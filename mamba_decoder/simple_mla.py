from typing import Optional, Literal

import torch
from torch import nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim).

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        # Compute RMS: sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


def precompute_freqs_cis(
    seq_len: int,
    head_dim: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Precompute frequency-based complex exponential values for rotary positional embeddings.

    Args:
        seq_len (int): Maximum sequence length.
        head_dim (int): Dimension of each attention head (must be even).
        theta (float): Base for rotary positional encoding. Defaults to 10000.0.
        device (Optional[torch.device]): Device to create tensors on.

    Returns:
        torch.Tensor: Precomputed complex exponential values of shape (seq_len, head_dim // 2).
    """
    assert head_dim % 2 == 0, "head_dim must be even for rotary embeddings"

    # Compute frequencies: 1 / (theta^(2i/dim))
    freqs = 1.0 / (
        theta
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )

    # Create positions
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)

    # Outer product: (seq_len, head_dim // 2)
    freqs = torch.outer(positions, freqs)

    # Convert to complex exponential: e^(i * freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (..., seq_len, head_dim).
        freqs_cis (torch.Tensor): Precomputed complex exponential values of shape (seq_len, head_dim // 2).

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied, same shape as input.
    """
    dtype = x.dtype

    # Reshape to complex: (..., seq_len, head_dim) -> (..., seq_len, head_dim//2)
    x_complex = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))

    # Reshape freqs_cis to match: (1, seq_len, 1, head_dim//2)
    # freqs_cis has shape (seq_len, head_dim//2)
    # We need to reshape it to broadcast: (1, seq_len, 1, head_dim//2)
    seq_len = freqs_cis.size(0)  # Sequence length from freqs_cis
    head_dim_half = freqs_cis.size(1)  # head_dim // 2 from freqs_cis
    freqs_cis = freqs_cis.view(1, seq_len, 1, head_dim_half)

    # Apply rotation: multiply complex numbers
    x_rotated = torch.view_as_real(x_complex * freqs_cis).flatten(3)

    return x_rotated.to(dtype)


class SimpleMLA(nn.Module):
    """
    Simple Multi-Head Latent Attention (MLA) implementation in pure PyTorch.

    This implementation captures the core ideas of MLA:
    - LoRA-based low-rank projections for Q and KV
    - Split attention into positional (RoPE) and non-positional components
    - KV caching for efficient inference
    - Two attention modes: naive (standard) and absorb (latent)

    Args:
        dim (int): Model dimension.
        n_heads (int): Number of attention heads.
        q_lora_rank (int): LoRA rank for query projection. If 0, uses full rank.
        kv_lora_rank (int): LoRA rank for key-value projection.
        qk_nope_head_dim (int): Dimension for non-positional Q/K projections.
        qk_rope_head_dim (int): Dimension for rotary-positional Q/K projections.
        v_head_dim (int): Dimension for value projections.
        max_batch_size (int): Maximum batch size for KV cache.
        max_seq_len (int): Maximum sequence length for KV cache.
        attn_impl (Literal["naive", "absorb"]): Attention implementation mode.
        rope_theta (float): Base for rotary positional encoding.
        mscale (float): Scaling factor for extended attention.
    """

    def __init__(
        self,
        dim: int = 2048,
        n_heads: int = 16,
        q_lora_rank: int = 0,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        max_batch_size: int = 8,
        max_seq_len: int = 4096,
        attn_impl: Literal["naive", "absorb"] = "absorb",
        rope_theta: float = 10000.0,
        mscale: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.attn_impl = attn_impl
        self.max_seq_len = max_seq_len

        # Query projection
        if q_lora_rank == 0:
            # Full rank projection
            self.wq = nn.Linear(dim, n_heads * self.qk_head_dim, bias=False)
        else:
            # LoRA projection: W = W_b @ (norm(W_a @ x))
            self.wq_a = nn.Linear(dim, q_lora_rank, bias=False)
            self.q_norm = RMSNorm(q_lora_rank)
            self.wq_b = nn.Linear(q_lora_rank, n_heads * self.qk_head_dim, bias=False)

        # Key-Value projection (always uses LoRA)
        # First projects to kv_lora_rank + qk_rope_head_dim
        self.wkv_a = nn.Linear(dim, kv_lora_rank + qk_rope_head_dim, bias=False)
        self.kv_norm = RMSNorm(kv_lora_rank)
        # Then projects from kv_lora_rank to (qk_nope_head_dim + v_head_dim) per head
        self.wkv_b = nn.Linear(
            kv_lora_rank,
            n_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
        )

        # Output projection
        self.wo = nn.Linear(n_heads * v_head_dim, dim, bias=False)

        # Attention scaling
        self.softmax_scale = self.qk_head_dim**-0.5
        if mscale != 1.0:
            self.softmax_scale = self.softmax_scale * mscale * mscale

        # Precompute rotary embeddings
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(max_seq_len, qk_rope_head_dim, rope_theta),
            persistent=False,
        )

        # KV cache buffers
        if attn_impl == "naive":
            self.register_buffer(
                "k_cache",
                torch.zeros(
                    max_batch_size,
                    max_seq_len,
                    n_heads,
                    self.qk_head_dim,
                ),
                persistent=False,
            )
            self.register_buffer(
                "v_cache",
                torch.zeros(
                    max_batch_size,
                    max_seq_len,
                    n_heads,
                    v_head_dim,
                ),
                persistent=False,
            )
        else:  # absorb mode
            self.register_buffer(
                "kv_cache",
                torch.zeros(
                    max_batch_size,
                    max_seq_len,
                    kv_lora_rank,
                ),
                persistent=False,
            )
            self.register_buffer(
                "pe_cache",
                torch.zeros(
                    max_batch_size,
                    max_seq_len,
                    qk_rope_head_dim,
                ),
                persistent=False,
            )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for SimpleMLA.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            start_pos (int): Starting position in the sequence for caching. Defaults to 0.
            mask (Optional[torch.Tensor]): Attention mask of shape (seq_len, seq_len).
                Values should be 0 for valid positions and -inf for masked positions.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim).
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen

        # Query projection
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))

        # Reshape and split Q into positional and non-positional parts
        q = q.view(bsz, seqlen, self.n_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # Apply rotary positional embedding to Q
        q_pe = apply_rotary_emb(q_pe, self.freqs_cis[start_pos:end_pos])

        # Key-Value projection
        kv = self.wkv_a(x)
        kv_latent, k_pe = torch.split(
            kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        # Apply rotary positional embedding to K (for positional part)
        k_pe = apply_rotary_emb(
            k_pe.unsqueeze(2), self.freqs_cis[start_pos:end_pos]
        )  # Add head dimension: (bsz, seqlen, 1, qk_rope_head_dim)

        if self.attn_impl == "naive":
            # Standard attention: compute full K and V
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv = self.wkv_b(self.kv_norm(kv_latent))
            kv = kv.view(
                bsz,
                seqlen,
                self.n_heads,
                self.qk_nope_head_dim + self.v_head_dim,
            )
            k_nope, v = torch.split(
                kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
            )
            k = torch.cat(
                [k_nope, k_pe.expand(-1, -1, self.n_heads, -1)],
                dim=-1,
            )

            # Update cache
            self.k_cache[:bsz, start_pos:end_pos] = k
            self.v_cache[:bsz, start_pos:end_pos] = v

            # Compute attention scores
            scores = (
                torch.einsum(
                    "bshd,bthd->bsht",
                    q,
                    self.k_cache[:bsz, :end_pos],
                )
                * self.softmax_scale
            )

            # Apply mask if provided
            if mask is not None:
                scores = scores + mask.unsqueeze(1)

            # Softmax and apply to values
            scores = scores.softmax(dim=-1)
            x = torch.einsum(
                "bsht,bthd->bshd",
                scores,
                self.v_cache[:bsz, :end_pos],
            )
        else:
            # Absorb mode: use latent KV cache
            kv_latent_norm = self.kv_norm(kv_latent)

            # Update cache
            self.kv_cache[:bsz, start_pos:end_pos] = kv_latent_norm
            self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)

            # Get weight matrix for projection
            wkv_b = self.wkv_b.weight.view(self.n_heads, -1, self.kv_lora_rank)

            # Project Q_nope through weight matrix
            q_nope_proj = torch.einsum(
                "bshd,hdc->bshc",
                q_nope,
                wkv_b[:, : self.qk_nope_head_dim],
            )

            # Compute attention scores: non-positional + positional
            scores = (
                torch.einsum(
                    "bshc,btc->bsht",
                    q_nope_proj,
                    self.kv_cache[:bsz, :end_pos],
                )
                + torch.einsum(
                    "bshr,btr->bsht",
                    q_pe,
                    self.pe_cache[:bsz, :end_pos],
                )
            ) * self.softmax_scale

            # Apply mask if provided
            if mask is not None:
                scores = scores + mask.unsqueeze(1)

            # Softmax and apply to latent cache
            scores = scores.softmax(dim=-1)
            x = torch.einsum(
                "bsht,btc->bshc",
                scores,
                self.kv_cache[:bsz, :end_pos],
            )

            # Project back to value space
            x = torch.einsum(
                "bshc,hdc->bshd",
                x,
                wkv_b[:, -self.v_head_dim :],
            )

        # Output projection
        x = self.wo(x.flatten(2))
        return x


# # Example usage
# if __name__ == "__main__":
#     # Create model
#     model = SimpleMLA(
#         dim=2048,
#         n_heads=16,
#         q_lora_rank=0,  # Use full rank for Q
#         kv_lora_rank=512,
#         qk_nope_head_dim=128,
#         qk_rope_head_dim=64,
#         v_head_dim=128,
#         max_batch_size=8,
#         max_seq_len=4096,
#         attn_impl="absorb",  # Use latent attention
#     )

#     # Test input
#     batch_size = 2
#     seq_len = 1024
#     x = torch.randn(batch_size, seq_len, 2048)
#     start_pos = 0
#     mask = None

#     # Forward pass
#     out = model(x, start_pos=start_pos, mask=mask)

#     print(f"Input shape: {x.shape}")
#     print(f"Output shape: {out.shape}")
#     print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
#     print(f"Output stats - mean: {out.mean().item():.4f}, std: {out.std().item():.4f}")

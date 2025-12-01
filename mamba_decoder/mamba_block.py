"""
Todo
- Add positional encoding
- Add ROPE
- NOPE option as well
"""

from typing import Literal, Optional

import torch
from mambapy.mamba import Mamba, MambaConfig
from open_kimi.moe import MoE
from pydantic import BaseModel
from torch import Tensor, nn

from simple_mla import SimpleMLA


class MoEConfig(BaseModel):
    """
    Configuration for Mixture of Experts (MoE) layer.

    This configuration class defines the parameters needed to initialize a MoE layer,
    including the number of experts, activation strategy, and adaptive bias settings.

    Args:
        dim (int): Model dimension. Defaults to 512.
        n_experts (int): Number of expert networks in the MoE layer. Defaults to 6.
        n_activated (int): Number of experts to activate per token. Defaults to 8.
        expert_inter_dim (Optional[int]): Intermediate dimension for expert networks.
            If None, uses default dimension. Defaults to None.
        shared_expert_inter_dim (Optional[int]): Intermediate dimension for shared expert.
            If None, uses default dimension. Defaults to None.
        use_adaptive_bias (bool): Whether to use adaptive bias in expert selection.
            Defaults to True.
        bias_update_rate (float): Learning rate for updating adaptive bias. Defaults to 0.01.
    """
    dim: int = 512
    n_experts: int = (6,)
    n_activated: int = (8,)
    expert_inter_dim: Optional[int] = (None,)
    shared_expert_inter_dim: Optional[int] = (None,)
    use_adaptive_bias: bool = (True,)
    bias_update_rate: float = (0.01,)


class MambaDecoderBlock(nn.Module):
    """
    A decoder block combining Mamba state space model with Mixture of Experts (MoE).

    This block implements a transformer-like decoder architecture that uses:
    - Mamba for efficient sequence modeling with state space models
    - MoE for parameter-efficient feed-forward processing
    - RMSNorm for normalization

    The forward pass follows a residual structure:
    1. Normalize input and apply Mamba, add residual
    2. Normalize Mamba output and apply MoE, add residual

    Args:
        dim (int): Model dimension. Defaults to 512.
        mamba_config (MambaConfig): Configuration for the Mamba layer. If None, uses default.
        moe_config (MoEConfig): Configuration for the MoE layer. If None, uses default.

    Example:
        >>> mamba_config = MambaConfig(d_model=512, n_layers=1)
        >>> moe_config = MoEConfig(dim=512, n_experts=6, n_activated=2)
        >>> block = MambaDecoderBlock(dim=512, mamba_config=mamba_config, moe_config=moe_config)
        >>> x = torch.randn(2, 128, 512)
        >>> out = block(x)
    """

    def __init__(
        self,
        dim: int = 512,
        mamba_config: MambaConfig = None,
        moe_config: MoEConfig = None,
    ):
        super().__init__()

        self.mamba = Mamba(
            config=mamba_config,
        )

        self.moe = MoE(
            dim=moe_config.dim,
            n_experts=moe_config.n_experts,
            n_activated=moe_config.n_activated,
            expert_inter_dim=moe_config.expert_inter_dim,
            shared_expert_inter_dim=moe_config.shared_expert_inter_dim,
            use_adaptive_bias=moe_config.use_adaptive_bias,
            bias_update_rate=moe_config.bias_update_rate,
        )

        self.norm = nn.RMSNorm(dim)

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the MambaDecoderBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim).
        """
        residual = x

        normed = self.norm(residual)

        mamba_out = self.mamba(normed) + residual

        # Then MoE
        normed_again = self.norm(mamba_out)

        # Then MoE
        moe_out = self.moe(normed_again) + mamba_out

        return moe_out


class MLABlock(nn.Module):
    """
    A decoder block combining Multi-Head Latent Attention (MLA) with Mixture of Experts (MoE).

    This block implements a transformer-like decoder architecture that uses:
    - SimpleMLA for efficient attention with LoRA-based projections and rotary embeddings
    - MoE for parameter-efficient feed-forward processing
    - RMSNorm for normalization

    The forward pass follows a residual structure:
    1. Normalize input and apply MLA attention, add residual
    2. Normalize MLA output and apply MoE, add residual

    Args:
        dim (int): Model dimension. Defaults to 2048.
        n_heads (int): Number of attention heads. Defaults to 16.
        q_lora_rank (int): LoRA rank for query projection. If 0, uses full rank. Defaults to 0.
        kv_lora_rank (int): LoRA rank for key-value projection. Defaults to 512.
        qk_nope_head_dim (int): Dimension for non-positional Q/K projections. Defaults to 128.
        qk_rope_head_dim (int): Dimension for rotary-positional Q/K projections. Defaults to 64.
        v_head_dim (int): Dimension for value projections. Defaults to 128.
        max_batch_size (int): Maximum batch size for KV cache. Defaults to 8.
        max_seq_len (int): Maximum sequence length for KV cache. Defaults to 4096.
        attn_impl (Literal["naive", "absorb"]): Attention implementation mode.
            "naive" uses standard attention, "absorb" uses latent attention. Defaults to "absorb".
        rope_theta (float): Base for rotary positional encoding. Defaults to 10000.0.
        mscale (float): Scaling factor for extended attention. Defaults to 1.0.
        n_experts (int): Number of expert networks in the MoE layer. Defaults to 12.
        n_activated (int): Number of experts to activate per token. Defaults to 4.
        expert_inter_dim (Optional[int]): Intermediate dimension for expert networks.
            If None, uses default dimension. Defaults to None.
        shared_expert_inter_dim (Optional[int]): Intermediate dimension for shared expert.
            If None, uses default dimension. Defaults to None.
        use_adaptive_bias (bool): Whether to use adaptive bias in expert selection. Defaults to True.
        bias_update_rate (float): Learning rate for updating adaptive bias. Defaults to 0.01.

    Example:
        >>> block = MLABlock(dim=2048, n_heads=16, max_seq_len=4096)
        >>> x = torch.randn(2, 1024, 2048)
        >>> out = block(x, start_pos=0, mask=None)
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
        n_experts: int = 12,
        n_activated: int = 4,
        expert_inter_dim: Optional[int] = None,
        shared_expert_inter_dim: Optional[int] = None,
        use_adaptive_bias: bool = True,
        bias_update_rate: float = 0.01,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.attn_impl = attn_impl
        self.rope_theta = rope_theta
        self.mscale = mscale
        self.n_experts = n_experts
        self.n_activated = n_activated
        self.expert_inter_dim = expert_inter_dim
        self.shared_expert_inter_dim = shared_expert_inter_dim
        self.use_adaptive_bias = use_adaptive_bias
        self.bias_update_rate = bias_update_rate

        self.mla = SimpleMLA(
            dim=dim,
            n_heads=n_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            attn_impl=attn_impl,
            rope_theta=rope_theta,
            mscale=mscale,
        )

        # MoE
        self.moe = MoE(
            dim=dim,
            n_experts=n_experts,
            n_activated=n_activated,
            expert_inter_dim=expert_inter_dim,
            shared_expert_inter_dim=shared_expert_inter_dim,
            use_adaptive_bias=use_adaptive_bias,
            bias_update_rate=bias_update_rate,
        )

        self.norm = nn.RMSNorm(dim)

    def forward(
        self,
        x: Tensor,
        start_pos: int = 0,
        mask: Optional[torch.Tensor] = None,
    ) -> Tensor:
        """
        Forward pass through the MLABlock.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, dim).
            start_pos (int): Starting position in the sequence for attention caching.
                Used for efficient incremental decoding. Defaults to 0.
            mask (Optional[torch.Tensor]): Attention mask of shape (seq_len, seq_len).
                Values should be 0 for valid positions and -inf for masked positions.
                Defaults to None.

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, dim).
        """
        residual = x

        # Normalize and apply MLA
        first = self.mla(self.norm(residual), start_pos, mask) + residual

        # Then MoE
        second = self.moe(self.norm(first)) + first

        return second


class MambaDecoder(nn.Module):
    """
    A full decoder model combining Mamba blocks with optional MLA attention.

    This model implements a complete language model decoder architecture with:
    - Token embeddings
    - Optional post-embedding normalization
    - Stack of MambaDecoderBlocks for sequence modeling
    - Optional MLABlock after Mamba blocks for attention-based refinement
    - Output head for vocabulary prediction

    The architecture allows for flexible configurations:
    - Pure Mamba-based decoding (post_mamba_mla_block=False)
    - Hybrid Mamba + MLA decoding (post_mamba_mla_block=True)

    Args:
        heads (int): Number of attention heads for MLA block. Defaults to 64.
        vocab_size (int): Size of the vocabulary. Defaults to 50000.
        max_seq_len (int): Maximum sequence length. Defaults to 512.
        depth (int): Number of MambaDecoderBlocks to stack. Defaults to 6.
        dim (int): Model dimension. Defaults to 512.
        post_embed_norm (bool): Whether to apply normalization after embedding.
            Defaults to True.
        mamba_config (MambaConfig): Configuration for Mamba layers. If None, uses default.
        moe_config (MoEConfig): Configuration for MoE layers. If None, uses default.
        post_mamba_mla_block (bool): Whether to apply MLABlock after Mamba blocks.
            If True, adds attention-based refinement. Defaults to False.

    Example:
        >>> mamba_config = MambaConfig(d_model=512, n_layers=1)
        >>> moe_config = MoEConfig(dim=512, n_experts=6, n_activated=2)
        >>> model = MambaDecoder(
        ...     vocab_size=32000,
        ...     max_seq_len=128,
        ...     depth=6,
        ...     dim=512,
        ...     heads=64,
        ...     mamba_config=mamba_config,
        ...     moe_config=moe_config,
        ...     post_mamba_mla_block=True
        ... )
        >>> x = torch.randint(0, 32000, (2, 128))
        >>> out = model(x, start_pos=0, mask=None)
    """

    def __init__(
        self,
        heads: int = 64,
        vocab_size: int = 50000,
        max_seq_len: int = 512,
        depth: int = 6,
        dim: int = 512,
        post_embed_norm: bool = True,
        mamba_config: MambaConfig = None,
        moe_config: MoEConfig = None,
        post_mamba_mla_block: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.depth = depth
        self.post_embed_norm = post_embed_norm
        self.post_mamba_mla_block = post_mamba_mla_block

        self.embedding = nn.Embedding(vocab_size, dim)

        self.layers = nn.ModuleList(
            [
                MambaDecoderBlock(
                    dim=dim,
                    mamba_config=mamba_config,
                    moe_config=moe_config,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.RMSNorm(dim)

        self.head = nn.Sequential(nn.RMSNorm(dim), nn.Linear(dim, vocab_size))

        self.mla_block = MLABlock(
            dim=dim,
            n_heads=heads,
            max_seq_len=max_seq_len,
            n_experts=moe_config.n_experts,
            n_activated=moe_config.n_activated,
            expert_inter_dim=moe_config.expert_inter_dim,
            shared_expert_inter_dim=moe_config.shared_expert_inter_dim,
            use_adaptive_bias=moe_config.use_adaptive_bias,
            bias_update_rate=moe_config.bias_update_rate,
        )

    def forward(
        self,
        x,
        start_pos: int = 0,
        mask: Optional[torch.Tensor] = None,
    ) -> Tensor:
        """
        Forward pass through the MambaDecoder.

        Args:
            x: Input token indices of shape (batch_size, seq_len).
            start_pos (int): Starting position in the sequence for attention caching.
                Used for efficient incremental decoding. Defaults to 0.
            mask (Optional[torch.Tensor]): Attention mask of shape (seq_len, seq_len).
                Values should be 0 for valid positions and -inf for masked positions.
                Only used if post_mamba_mla_block is True. Defaults to None.

        Returns:
            Tensor: Output logits of shape (batch_size, seq_len, vocab_size).
        """
        x = self.embedding(x)

        if self.post_embed_norm is True:
            x = self.norm(x)

        # Then the layers
        for layer in self.layers:
            x = layer(x)

        if self.post_mamba_mla_block is True:
            x = self.mla_block(x, start_pos, mask)

        return self.head(x)


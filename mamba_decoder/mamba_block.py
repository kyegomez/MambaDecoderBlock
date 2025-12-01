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
    dim: int = 512
    n_experts: int = (6,)
    n_activated: int = (8,)
    expert_inter_dim: Optional[int] = (None,)
    shared_expert_inter_dim: Optional[int] = (None,)
    use_adaptive_bias: bool = (True,)
    bias_update_rate: float = (0.01,)


class MambaDecoderBlock(nn.Module):
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
        residual = x

        normed = self.norm(residual)

        mamba_out = self.mamba(normed) + residual

        # Then MoE
        normed_again = self.norm(mamba_out)

        # Then MoE
        moe_out = self.moe(normed_again) + mamba_out

        return moe_out


class MLABlock(nn.Module):
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
        residual = x

        # Normalize and apply MLA
        first = self.mla(self.norm(residual), start_pos, mask) + residual

        # Then MoE
        second = self.moe(self.norm(first)) + first

        return second


class MambaDecoder(nn.Module):
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
        x = self.embedding(x)

        if self.post_embed_norm is True:
            x = self.norm(x)

        # Then the layers
        for layer in self.layers:
            x = layer(x)

        if self.post_mamba_mla_block is True:
            x = self.mla_block(x, start_pos, mask)

        return self.head(x)


# # Test
# x = torch.randint(0, 32000, (2, 128))

# model = MambaDecoder(
#     vocab_size=32000,
#     max_seq_len=128,
#     depth=6,
#     dim=512,
#     post_embed_norm=True,
#     mamba_config=MambaConfig(
#         d_model=512,
#         n_layers=1,
#     ),
#     moe_config=MoEConfig(
#         dim=512,
#         n_experts=6,
#         n_activated=2,
#         expert_inter_dim=None,
#         shared_expert_inter_dim=None,
#         use_adaptive_bias=True,
#         bias_update_rate=0.01,
#     ),
# )

# out = model(x)
# print(out.shape)
# print(out)

# # Mla Test
# x = torch.randn(1, 1024, 2048)
# start_pos = 0
# mask = None

# model = MLABlock()
# out = model(x, start_pos, mask)
# print(out.shape)
# print(out)


# Post mamba mla test
# Test
x = torch.randint(0, 32000, (2, 128))

model = MambaDecoder(
    vocab_size=32000,
    max_seq_len=128,
    depth=6,
    dim=512,
    heads=64,
    post_embed_norm=True,
    mamba_config=MambaConfig(
        d_model=512,
        n_layers=1,
    ),
    moe_config=MoEConfig(
        dim=512,
        n_experts=6,
        n_activated=2,
        expert_inter_dim=None,
        shared_expert_inter_dim=None,
        use_adaptive_bias=True,
        bias_update_rate=0.01,
    ),
    post_mamba_mla_block=True,
)

out = model(x, start_pos=0, mask=None)
print(out.shape)
print(out)

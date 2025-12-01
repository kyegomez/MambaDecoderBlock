import torch
from mamba_decoder.mamba_block import MambaConfig, MoEConfig, MambaDecoder

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

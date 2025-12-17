# models/improve_transformer.py
import torch
import torch.nn as nn


class ImproveTransformerEncoderState(nn.Module):
    """
    Improved Transformer Encoder for UAV State Representation
    """

    def __init__(self, seq_len, config):
        super().__init__()
        self.seq_len = seq_len
        d_model = config.get('embed_dim', 20)
        num_layers = config.get('num_layers', 2)
        expand_ratio = config.get('ff_expand_ratio', 4)

        # input projection
        self.input_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )

        # state token and position encoding
        self.state_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_encoding = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))

        # improved transformer blocks
        self.layers = nn.ModuleList([
            ImproveTransformerBlock(d_model, expand_ratio)
            for _ in range(num_layers)
        ])

        # output normalization
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, obs_seq):
        batch_size = obs_seq.size(0)

        # input projection
        x = self.input_proj(obs_seq)

        # add state token
        st = self.state_token.expand(batch_size, -1, -1)
        x = torch.cat([st, x], dim=1)
        x = x + self.pos_encoding

        # improved transformer processing
        for layer in self.layers:
            x = layer(x)

        # output state feature
        state_feat = self.output_norm(x[:, 0, :])
        return state_feat


class ImproveTransformerBlock(nn.Module):
    """Improved Transformer Block"""

    def __init__(self, d_model, expand_ratio=4):
        super().__init__()

        # Pre-Norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # self-attention with single head (for 20-dim data)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=1,  # keep single head
            batch_first=True,
            dropout=0.1
        )

        # improved FFN - use GELU and slight expansion
        ffn_dim = d_model * expand_ratio
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(0.1)
        )

        # residual scaling
        self.residual_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # Pre-Norm + self-attention + residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out * self.residual_scale

        # Pre-Norm + FFN + residual
        x_norm2 = self.norm2(x)
        ffn_out = self.ffn(x_norm2)
        x = x + ffn_out * self.residual_scale

        return x
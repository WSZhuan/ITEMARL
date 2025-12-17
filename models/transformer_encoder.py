# models/transformer_encoder.py
import torch
import torch.nn as nn

class TransformerEncoderState(nn.Module):
    """
    Transformer encoder for observation sequences.
    - state_token: learnable [1,1,embed_dim]
    - pos_encoding: learnable [1, N+1, embed_dim]
    - L Transformer blocks with SSA, LN, FFN
    - embed_dim = config['embed_dim'] (default=20)
    - num_layers = config['num_layers'] (default=2)
    - expand_ratio = config['ff_expand_ratio'] (default=4)
    """
    def __init__(self, seq_len, config):
        super().__init__()
        self.seq_len = seq_len
        d_model = config.get('embed_dim', 20)
        num_layers = config.get('num_layers', 2)
        expand_ratio = config.get('ff_expand_ratio', 4)
        # State token and positional encoding
        self.state_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_encoding = nn.Parameter(torch.zeros(1, seq_len+1, d_model))
        # Input projection
        self.input_proj = nn.Linear(d_model, d_model)
        # Build Transformer blocks
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            block = nn.ModuleDict({
                'ln1': nn.LayerNorm(d_model),
                'attn': nn.MultiheadAttention(embed_dim=d_model, num_heads=1, batch_first=True),
                'ln2': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, expand_ratio*d_model),
                    nn.ReLU(),
                    nn.Linear(expand_ratio*d_model, d_model)
                ),
            })
            self.layers.append(block)

    def forward(self, obs_seq):
        """
        obs_seq: [batch, N, embed_dim]
        returns: state feature [batch, embed_dim]
        """
        batch_size = obs_seq.size(0)
        # project inputs if needed
        x = self.input_proj(obs_seq)
        # prepend state token
        st = self.state_token.expand(batch_size, -1, -1)
        x = torch.cat([st, x], dim=1)  # [batch, N+1, d_model]
        # add positional encoding
        x = x + self.pos_encoding
        # Transformer blocks
        for layer in self.layers:
            # Self-Attention
            y = layer['ln1'](x)
            attn_out, _ = layer['attn'](y, y, y)
            x = x + attn_out
            # Feedforward
            y = layer['ln2'](x)
            x = x + layer['ffn'](y)
        # state token output
        state_feat = x[:, 0, :]
        return state_feat

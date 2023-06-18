import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, forward_expansion * embed_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_dim, embed_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(query, key, value, attn_mask=mask)[0]
        x = self.norm1(attention + query)
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)
        return out

class Encoder(nn.Module):
    def __init__(self, 
                 source_vocab_size, 
                 embed_dim, 
                 num_layers, 
                 num_heads, 
                 device, 
                 dropout, 
                 max_length, 
                 forward_expansion):
        super(Encoder, self).__init__()

        self.embed_dim = embed_dim
        self.device = device
        self.word_embedding = nn.Embedding(source_vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim,
                    num_heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out

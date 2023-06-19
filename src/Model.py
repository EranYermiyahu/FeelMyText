import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerECT(nn.Module):
    def __init__(self, input_dim, n_labels, hidden_dim, num_layers, num_heads, dropout):
        super(TransformerECT, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout)

        encoder_layer = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, n_labels)

    def forward(self, src, mask):
        embedded = self.embedding(src)
        encoded = self.positional_encoding(embedded)

        output = self.transformer_encoder(encoded, src_key_padding_mask=mask.float().transpose(0, 1))

        output = self.layer_norm(output)
        pooled = output.mean(dim=0)  # Average pooling over the sequence dimension
        logits = self.fc(pooled)

        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))
        pe = torch.zeros(max_len, hidden_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# class TransformerBlock(nn.Module):
#     def __init__(self, embed_dim, num_heads, dropout, forward_expansion):
#         super(TransformerBlock, self).__init__()
#         self.attention = nn.MultiheadAttention(embed_dim, num_heads)
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.norm2 = nn.LayerNorm(embed_dim)
#
#         self.feed_forward = nn.Sequential(
#             nn.Linear(embed_dim, forward_expansion * embed_dim),
#             nn.ReLU(),
#             nn.Linear(forward_expansion * embed_dim, embed_dim),
#         )
#
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, value, key, query, mask):
#         attention = self.attention(query, key, value, attn_mask=mask)[0]
#         x = self.norm1(attention + query)
#         forward = self.feed_forward(x)
#         out = self.norm2(forward + x)
#         return out
#
#
# class Encoder(nn.Module):
#     def __init__(self,
#                  source_vocab_size,
#                  embed_dim,
#                  num_layers,
#                  num_heads,
#                  device,
#                  dropout,
#                  max_length,
#                  forward_expansion):
#         super(Encoder, self).__init__()
#
#         self.embed_dim = embed_dim
#         self.device = device
#         self.word_embedding = nn.Embedding(source_vocab_size, embed_dim)
#         self.position_embedding = nn.Embedding(max_length, embed_dim)
#
#         self.layers = nn.ModuleList(
#             [
#                 TransformerBlock(
#                     embed_dim,
#                     num_heads,
#                     dropout=dropout,
#                     forward_expansion=forward_expansion,
#                 )
#                 for _ in range(num_layers)
#             ]
#         )
#
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x, mask):
#         N, seq_length = x.shape
#         positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
#         out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
#
#         for layer in self.layers:
#             out = layer(out, out, out, mask)
#
#         return out
#
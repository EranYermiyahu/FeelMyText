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
        embedded = self.embedding(src.transpose(0, 1))
        encoded = self.positional_encoding(embedded)
        encoded = self.layer_norm(encoded)

        output = self.transformer_encoder(encoded, src_key_padding_mask=mask.float())

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

import torch
import torch.nn as nn

from .layers import MultiHeadAttentionLayer, PositionwiseFeedforwardLayer


class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(
            hid_dim, pf_dim, dropout
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, source: torch.Tensor):
        _source, _ = self.self_attention(source, source, source)
        source = self.layer_norm(source + self.dropout(_source))

        _source = self.positionwise_feedforward(source)
        source = self.layer_norm(source + self.dropout(_source))

        return source


class Encoder(nn.Module):
    def __init__(
        self,
        n_frame: int,
        n_bin: int,
        cnn_channel: int,
        cnn_kernel: int,
        hid_dim: int,
        n_margin: int,
        n_layers: int,
        n_heads: int,
        pf_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.n_frame = n_frame
        self.n_bin = n_bin
        self.cnn_channel = cnn_channel
        self.cnn_kernel = cnn_kernel
        self.hid_dim = hid_dim
        self.n_proc = n_margin * 2 + 1
        self.cnn_dim = self.cnn_channel * (self.n_proc - (self.cnn_kernel - 1))

        self.conv = nn.Conv2d(1, self.cnn_channel, kernel_size=(1, self.cnn_kernel))
        self.tok_embedding_freq = nn.Linear(self.cnn_dim, hid_dim)
        self.pos_embedding_freq = nn.Embedding(n_bin, hid_dim)
        self.layers_freq = nn.ModuleList(
            [EncoderLayer(hid_dim, n_heads, pf_dim, dropout) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, spec: torch.Tensor):
        batch_size = spec.shape[0]

        spec = spec.unfold(2, self.n_proc, 1).permute(0, 2, 1, 3).contiguous()

        # CNN 1D
        spec_cnn = spec.reshape(
            batch_size * self.n_frame, self.n_bin, self.n_proc
        ).unsqueeze(1)
        spec_cnn = self.conv(spec_cnn).permute(0, 2, 1, 3).contiguous()

        spec_cnn_freq = spec_cnn.reshape(
            batch_size * self.n_frame, self.n_bin, self.cnn_dim
        )

        # embedding
        spec_emb_freq = self.tok_embedding_freq(spec_cnn_freq)

        # position coding
        pos_freq = (
            torch.arange(0, self.n_bin)
            .unsqueeze(0)
            .repeat(batch_size * self.n_frame, 1).to(spec.device)
        )

        # embedding
        scale_freq = torch.sqrt(torch.FloatTensor([self.hid_dim])).to(spec.device)
        spec_freq = self.dropout(
            (spec_emb_freq * scale_freq) + self.pos_embedding_freq(pos_freq)
        )

        # transformer encoder
        for layer_freq in self.layers_freq:
            spec_freq = layer_freq(spec_freq)
        spec_freq = spec_freq.reshape(
            batch_size, self.n_frame, self.n_bin, self.hid_dim
        )

        return spec_freq

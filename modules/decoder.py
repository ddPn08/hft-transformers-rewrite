import torch
import torch.nn as nn

from .encoder import EncoderLayer
from .layers import MultiHeadAttentionLayer, PositionwiseFeedforwardLayer


class DecoderLayerZero(nn.Module):
    def __init__(
        self,
        hid_dim: int,
        n_heads: int,
        pf_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hid_dim)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(
            hid_dim, pf_dim, dropout
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, source: torch.Tensor, target: torch.Tensor):
        _trg, attention = self.encoder_attention(target, source, source)
        target = self.layer_norm(target + self.dropout(_trg))

        _trg = self.positionwise_feedforward(target)
        target = self.layer_norm(target + self.dropout(_trg))

        return target, attention


class DecoderLayer(nn.Module):
    def __init__(
        self,
        hid_dim: int,
        n_heads: int,
        pf_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(
            hid_dim, pf_dim, dropout
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, source: torch.Tensor, target: torch.Tensor):
        _target, _ = self.self_attention(target, target, target)
        target = self.layer_norm(target + self.dropout(_target))

        _target, attention = self.encoder_attention(target, source, source)
        target = self.layer_norm(target + self.dropout(_target))

        _target = self.positionwise_feedforward(target)
        target = self.layer_norm(target + self.dropout(_target))

        return target, attention


class Decoder(nn.Module):
    def __init__(
        self,
        n_frame: int,
        n_bin: int,
        n_note: int,
        n_velocity: int,
        hid_dim: int,
        n_layers: int,
        n_heads: int,
        pf_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.n_frame = n_frame
        self.n_bin = n_bin
        self.n_note = n_note
        self.n_velocity = n_velocity
        self.hid_dim = hid_dim

        self.dropout = nn.Dropout(dropout)

        self.pos_embedding_freq = nn.Embedding(n_note, hid_dim)
        self.layer_zero_freq = DecoderLayerZero(hid_dim, n_heads, pf_dim, dropout)
        self.layers_freq = nn.ModuleList(
            [
                DecoderLayer(hid_dim, n_heads, pf_dim, dropout)
                for _ in range(n_layers - 1)
            ]
        )
        self.fc_onset_freq = nn.Linear(hid_dim, 1)
        self.fc_offset_freq = nn.Linear(hid_dim, 1)
        self.fc_mpe_freq = nn.Linear(hid_dim, 1)
        self.fc_velocity_freq = nn.Linear(hid_dim, self.n_velocity)

        self.pos_embedding_time = nn.Embedding(n_frame, hid_dim)
        self.layers_time = nn.ModuleList(
            [EncoderLayer(hid_dim, n_heads, pf_dim, dropout) for _ in range(n_layers)]
        )

        self.fc_onset_time = nn.Linear(hid_dim, 1)
        self.fc_offset_time = nn.Linear(hid_dim, 1)
        self.fc_mpe_time = nn.Linear(hid_dim, 1)
        self.fc_velocity_time = nn.Linear(hid_dim, self.n_velocity)

    def forward(self, spec: torch.Tensor):
        batch_size = spec.shape[0]
        spec = spec.reshape([batch_size * self.n_frame, self.n_bin, self.hid_dim])

        pos_freq = (
            torch.arange(0, self.n_note)
            .unsqueeze(0)
            .repeat(batch_size * self.n_frame, 1)
            .to(spec.device)
        )
        midi_freq = self.pos_embedding_freq(pos_freq)

        midi_freq, attention_freq = self.layer_zero_freq(spec, midi_freq)

        for layer_freq in self.layers_freq:
            midi_freq, attention_freq = layer_freq(spec, midi_freq)

        dim = attention_freq.shape
        attention_freq = attention_freq.reshape(
            [batch_size, self.n_frame, dim[1], dim[2], dim[3]]
        )

        output_onset_freq = self.fc_onset_freq(midi_freq).reshape(
            [batch_size, self.n_frame, self.n_note]
        )
        output_offset_freq = self.fc_offset_freq(midi_freq).reshape(
            [batch_size, self.n_frame, self.n_note]
        )
        output_mpe_freq = self.fc_mpe_freq(midi_freq).reshape(
            [batch_size, self.n_frame, self.n_note]
        )
        output_velocity_freq = self.fc_velocity_freq(midi_freq).reshape(
            [batch_size, self.n_frame, self.n_note, self.n_velocity]
        )

        midi_time = (
            midi_freq.reshape([batch_size, self.n_frame, self.n_note, self.hid_dim])
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape([batch_size * self.n_note, self.n_frame, self.hid_dim])
        )
        pos_time = (
            torch.arange(0, self.n_frame)
            .unsqueeze(0)
            .repeat(batch_size * self.n_note, 1)
            .to(spec.device)
        )
        scale_time = torch.sqrt(torch.FloatTensor([self.hid_dim])).to(spec.device)
        midi_time = self.dropout(
            (midi_time * scale_time) + self.pos_embedding_time(pos_time)
        )

        for layer_time in self.layers_time:
            midi_time = layer_time(midi_time)

        output_onset_time = (
            self.fc_onset_time(midi_time)
            .reshape([batch_size, self.n_note, self.n_frame])
            .permute(0, 2, 1)
            .contiguous()
        )
        output_offset_time = (
            self.fc_offset_time(midi_time)
            .reshape([batch_size, self.n_note, self.n_frame])
            .permute(0, 2, 1)
            .contiguous()
        )
        output_mpe_time = (
            self.fc_mpe_time(midi_time)
            .reshape([batch_size, self.n_note, self.n_frame])
            .permute(0, 2, 1)
            .contiguous()
        )
        output_velocity_time = (
            self.fc_velocity_time(midi_time)
            .reshape([batch_size, self.n_note, self.n_frame, self.n_velocity])
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        return (
            output_onset_freq,
            output_offset_freq,
            output_mpe_freq,
            output_velocity_freq,
            attention_freq,
            output_onset_time,
            output_offset_time,
            output_mpe_time,
            output_velocity_time,
        )


class DecoderPedal(nn.Module):
    def __init__(
        self,
        n_frame: int,
        n_bin: int,
        hid_dim: int,
        n_layers: int,
        n_heads: int,
        pf_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.n_frame = n_frame
        self.n_bin = n_bin
        self.hid_dim = hid_dim

        self.dropout = nn.Dropout(dropout)

        self.pos_embedding_freq = nn.Embedding(1, hid_dim)
        self.layer_zero_freq = DecoderLayerZero(hid_dim, n_heads, pf_dim, dropout)
        self.layers_freq = nn.ModuleList(
            [
                DecoderLayer(hid_dim, n_heads, pf_dim, dropout)
                for _ in range(n_layers - 1)
            ]
        )

        self.fc_onpedal_freq = nn.Linear(hid_dim, 1)
        self.fc_offpedal_freq = nn.Linear(hid_dim, 1)
        self.fc_mpe_pedal_freq = nn.Linear(hid_dim, 1)

        self.pos_embedding_time = nn.Embedding(n_frame, hid_dim)
        self.layers_time = nn.ModuleList(
            [EncoderLayer(hid_dim, n_heads, pf_dim, dropout) for _ in range(n_layers)]
        )

        self.fc_onpedal_time = nn.Linear(hid_dim, 1)
        self.fc_offpedal_time = nn.Linear(hid_dim, 1)
        self.fc_mpe_pedal_time = nn.Linear(hid_dim, 1)

    def forward(self, spec: torch.Tensor):
        batch_size = spec.shape[0]
        spec = spec.reshape([batch_size * self.n_frame, self.n_bin, self.hid_dim])

        pos_freq = (
            torch.arange(0, 1)
            .unsqueeze(0)
            .repeat(batch_size * self.n_frame, 1)
            .to(spec.device)
        )
        midi_freq = self.pos_embedding_freq(pos_freq)

        midi_freq, attention_freq = self.layer_zero_freq(spec, midi_freq)
        for layer_freq in self.layers_freq:
            midi_freq, attention_freq = layer_freq(spec, midi_freq)

        dim = attention_freq.shape
        attention_freq = attention_freq.reshape(
            [batch_size, self.n_frame, dim[1], dim[2], dim[3]]
        )
        output_onpedal_freq = self.fc_onpedal_freq(midi_freq).reshape(
            [batch_size, self.n_frame]
        )
        output_offpedal_freq = self.fc_offpedal_freq(midi_freq).reshape(
            [batch_size, self.n_frame]
        )
        output_mpe_pedal_freq = self.fc_mpe_pedal_freq(midi_freq).reshape(
            [batch_size, self.n_frame]
        )

        midi_time = (
            midi_freq.reshape([batch_size, self.n_frame, self.hid_dim])
            .contiguous()
            .reshape([batch_size, self.n_frame, self.hid_dim])
        )
        pos_time = (
            torch.arange(0, self.n_frame)
            .unsqueeze(0)
            .repeat(batch_size, 1)
            .to(spec.device)
        )
        scale_time = torch.sqrt(torch.FloatTensor([self.hid_dim])).to(spec.device)
        midi_time = self.dropout(
            (midi_time * scale_time) + self.pos_embedding_time(pos_time)
        )

        for layer_time in self.layers_time:
            midi_time = layer_time(midi_time)

        output_onpedal_time = (
            self.fc_onpedal_time(midi_time)
            .reshape([batch_size, self.n_frame])
            .contiguous()
        )
        output_offpedal_time = (
            self.fc_offpedal_time(midi_time)
            .reshape([batch_size, self.n_frame])
            .contiguous()
        )
        output_mpe_pedal_time = (
            self.fc_mpe_pedal_time(midi_time)
            .reshape([batch_size, self.n_frame])
            .contiguous()
        )

        return (
            output_onpedal_freq,
            output_offpedal_freq,
            output_mpe_pedal_freq,
            attention_freq,
            output_onpedal_time,
            output_offpedal_time,
            output_mpe_pedal_time,
        )

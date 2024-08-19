from typing import Literal

import torch
import torch.nn as nn
from pydantic import BaseModel

from .decoder import Decoder, DecoderPedal
from .encoder import Encoder


class TranscriberConfig(BaseModel):
    n_frame: int
    n_bin: int
    cnn_channel: int
    cnn_kernel: int
    hid_dim: int
    n_margin: int
    n_layers: int
    n_heads: int
    pf_dim: int
    dropout: float
    n_velocity: int
    n_note: int


class Transcriber(nn.Module):
    def __init__(
        self, params: TranscriberConfig, mode: Literal["note", "pedal"] = "note"
    ):
        super().__init__()
        self.encoder = Encoder(
            n_frame=params.n_frame,
            n_bin=params.n_bin,
            cnn_channel=params.cnn_channel,
            cnn_kernel=params.cnn_kernel,
            hid_dim=params.hid_dim,
            n_margin=params.n_margin,
            n_layers=params.n_layers,
            n_heads=params.n_heads,
            pf_dim=params.pf_dim,
            dropout=params.dropout,
        )
        if mode == "note":
            self.decoder = Decoder(
                n_frame=params.n_frame,
                n_bin=params.n_bin,
                n_note=params.n_note,
                n_velocity=params.n_velocity,
                hid_dim=params.hid_dim,
                n_layers=params.n_layers,
                n_heads=params.n_heads,
                pf_dim=params.pf_dim,
                dropout=params.dropout,
            )
        elif mode == "pedal":
            self.decoder = DecoderPedal(
                n_frame=params.n_frame,
                n_bin=params.n_bin,
                hid_dim=params.hid_dim,
                n_layers=params.n_layers,
                n_heads=params.n_heads,
                pf_dim=params.pf_dim,
                dropout=params.dropout,
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def forward(self, spec: torch.Tensor):
        enc_vector = self.encoder(spec)
        return self.decoder(enc_vector)

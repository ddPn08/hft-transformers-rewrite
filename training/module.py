from typing import Any, Literal

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule

from modules.transcriber import Transcriber


class TranscriberModule(LightningModule):
    def __init__(
        self,
        model: Transcriber,
        optimizer_class: Any,
        lr: float = 1e-4,
        mode: Literal["note", "pedal"] = "note",
    ):
        super().__init__()
        self.model = model
        self.optimizer_class = optimizer_class
        self.lr = lr
        self.mode = mode
        self.all_loss = []
        self.epoch_loss = []

        if mode == 'note':
            self.criterion_onset_A = nn.BCEWithLogitsLoss()
            self.criterion_offset_A = nn.BCEWithLogitsLoss()
            self.criterion_mpe_A = nn.BCEWithLogitsLoss()
            self.criterion_velocity_A = nn.CrossEntropyLoss()

            self.criterion_onset_B = nn.BCEWithLogitsLoss()
            self.criterion_offset_B = nn.BCEWithLogitsLoss()
            self.criterion_mpe_B = nn.BCEWithLogitsLoss()
            self.criterion_velocity_B = nn.CrossEntropyLoss()
        elif mode == 'pedal':
            self.criterion_onpedal_A = nn.BCEWithLogitsLoss()
            self.criterion_offpedal_A = nn.BCEWithLogitsLoss()
            self.criterion_mpe_pedal_A = nn.BCEWithLogitsLoss()

            self.criterion_onpedal_B = nn.BCEWithLogitsLoss()
            self.criterion_offpedal_B = nn.BCEWithLogitsLoss()
            self.criterion_mpe_pedal_B = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def forward(self, y: torch.Tensor):
        return self.model(y)

    def training_step(self, batch: torch.Tensor, _: int):
        if self.mode == "note":
            (
                input_spec,
                label_onset,
                label_offset,
                label_mpe,
                label_velocity,
            ) = batch

            (
                output_onset_A,
                output_offset_A,
                output_mpe_A,
                output_velocity_A,
                _,
                output_onset_B,
                output_offset_B,
                output_mpe_B,
                output_velocity_B,
            ) = self.model(input_spec)

            output_onset_A = output_onset_A.contiguous().view(-1)
            output_offset_A = output_offset_A.contiguous().view(-1)
            output_mpe_A = output_mpe_A.contiguous().view(-1)
            output_velocity_A_dim = output_velocity_A.shape[-1]
            output_velocity_A = output_velocity_A.contiguous().view(
                -1, output_velocity_A_dim
            )

            output_onset_B = output_onset_B.contiguous().view(-1)
            output_offset_B = output_offset_B.contiguous().view(-1)
            output_mpe_B = output_mpe_B.contiguous().view(-1)
            output_velocity_B_dim = output_velocity_B.shape[-1]
            output_velocity_B = output_velocity_B.contiguous().view(
                -1, output_velocity_B_dim
            )

            label_onset = label_onset.contiguous().view(-1)
            label_offset = label_offset.contiguous().view(-1)
            label_mpe = label_mpe.contiguous().view(-1)
            label_velocity = label_velocity.contiguous().view(-1)

            loss_onset_A = self.criterion_onset_A(output_onset_A, label_onset)
            loss_offset_A = self.criterion_offset_A(output_offset_A, label_offset)
            loss_mpe_A = self.criterion_mpe_A(output_mpe_A, label_mpe)
            loss_velocity_A = self.criterion_velocity_A(
                output_velocity_A, label_velocity
            )
            loss_A = loss_onset_A + loss_offset_A + loss_mpe_A + loss_velocity_A

            loss_onset_B = self.criterion_onset_B(output_onset_B, label_onset)
            loss_offset_B = self.criterion_offset_B(output_offset_B, label_offset)
            loss_mpe_B = self.criterion_mpe_B(output_mpe_B, label_mpe)
            loss_velocity_B = self.criterion_velocity_B(
                output_velocity_B, label_velocity
            )
            loss_B = loss_onset_B + loss_offset_B + loss_mpe_B + loss_velocity_B

            loss = loss_A + loss_B

            self.all_loss.append(loss.item())
            self.epoch_loss.append(loss.item())
            self.log("train_loss", loss)

            return loss
        elif self.mode == "pedal":
            (
                input_spec,
                label_onpedal,
                label_offpedal,
                label_mpe_pedal,
            ) = batch

            (
                output_onpedal_A,
                output_offpedal_A,
                output_mpe_pedal_A,
                _,
                output_onpedal_B,
                output_offpedal_B,
                output_mpe_pedal_B,
            ) = self.model(input_spec)

            output_onpedal_A = output_onpedal_A.contiguous().view(-1)
            output_offpedal_A = output_offpedal_A.contiguous().view(-1)
            output_mpe_pedal_A = output_mpe_pedal_A.contiguous().view(-1)

            output_onpedal_B = output_onpedal_B.contiguous().view(-1)
            output_offpedal_B = output_offpedal_B.contiguous().view(-1)
            output_mpe_pedal_B = output_mpe_pedal_B.contiguous().view(-1)

            label_onpedal = label_onpedal.contiguous().view(-1)
            label_offpedal = label_offpedal.contiguous().view(-1)
            label_mpe_pedal = label_mpe_pedal.contiguous().view(-1)

            loss_onpedal_A = self.criterion_onpedal_A(output_onpedal_A, label_onpedal)
            loss_offpedal_A = self.criterion_offpedal_A(
                output_offpedal_A, label_offpedal
            )
            loss_mpe_pedal_A = self.criterion_mpe_pedal_A(
                output_mpe_pedal_A, label_mpe_pedal
            )
            loss_A = loss_onpedal_A + loss_offpedal_A + loss_mpe_pedal_A

            loss_onpedal_B = self.criterion_onpedal_B(output_onpedal_B, label_onpedal)
            loss_offpedal_B = self.criterion_offpedal_B(
                output_offpedal_B, label_offpedal
            )
            loss_mpe_pedal_B = self.criterion_mpe_pedal_B(
                output_mpe_pedal_B, label_mpe_pedal
            )
            loss_B = loss_onpedal_B + loss_offpedal_B + loss_mpe_pedal_B

            loss = loss_A + loss_B

            self.all_loss.append(loss.item())
            self.epoch_loss.append(loss.item())
            self.log("train_loss", loss)

            return loss
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def training_epoch_start(self, _):
        self.epoch_loss = []

    def configure_optimizers(self):
        return self.optimizer_class(self.parameters(), lr=self.lr)

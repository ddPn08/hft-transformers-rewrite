import os
from typing import Literal

import fire
import numpy as np
import torch
import torch.utils.data as data
from lightning.fabric.plugins.precision.precision import _PRECISION_INPUT
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)

from modules.transcriber import Transcriber, TranscriberConfig
from training.config import DatasetConfig, ModelConfig
from training.dataset import Dataset
from training.module import TranscriberModule

torch.set_float32_matmul_precision("medium")


class MyProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, pl_module):
        items = super().get_metrics(trainer, pl_module)
        items["loss"] = pl_module.all_loss[-1] if pl_module.all_loss else float("nan")
        items["all_loss_mean"] = np.mean(pl_module.all_loss or float("nan"))
        items["epoch_loss_mean"] = np.mean(pl_module.epoch_loss or float("nan"))
        return items


def main(
    dataset_dir: str = "./maestro-v3.0.0-preprocessed",
    output_dir: str = "./output",
    mode: Literal["note", "pedal"] = "note",
    accelerator: str = "gpu",
    devices: str = "0,",
    max_train_epochs: int = 100,
    precision: _PRECISION_INPUT = 32,
    batch_size: int = 1,
    num_workers: int = 1,
    logger: str = "none",
    logger_name: str = "training",
    logger_project: str = "hft-transformer",
):
    with open(os.path.join(dataset_dir, "config.json"), "r") as f:
        config: DatasetConfig = DatasetConfig.model_validate_json(f.read())

    dataset = Dataset(dir=dataset_dir, mode=mode)
    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
    )

    val_dataset = Dataset(dir=dataset_dir, split="validation", mode=mode)
    val_dataloader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn
    )

    params = TranscriberConfig(
        n_frame=config.input.num_frame,
        n_bin=config.feature.n_bins,
        cnn_channel=4,
        cnn_kernel=5,
        hid_dim=256,
        n_margin=config.input.margin_b,
        n_layers=3,
        n_heads=4,
        pf_dim=512,
        dropout=0.1,
        n_velocity=config.midi.num_velocity,
        n_note=config.midi.num_notes,
    )
    model_config = ModelConfig(
        mode=mode,
        params=params,
        feature=config.feature,
        input=config.input,
        midi=config.midi,
    )
    transcriber = Transcriber(params, mode=mode)
    module = TranscriberModule(transcriber, torch.optim.Adam, mode=mode)

    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        f.write(model_config.model_dump_json(indent=4))

    if logger == "wandb":
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(
            name=logger_name,
            project=logger_project,
        )
    else:
        logger = None

    callbacks = [
        MyProgressBar(),
        ModelCheckpoint(every_n_epochs=1, dirpath=checkpoint_dir, save_top_k=10, mode="min", monitor="val_loss"),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = Trainer(
        logger=logger,
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_train_epochs,
        log_every_n_steps=1,
        callbacks=callbacks,
        precision=precision,
    )
    trainer.fit(module, dataloader, val_dataloader)


if __name__ == "__main__":
    fire.Fire(main)

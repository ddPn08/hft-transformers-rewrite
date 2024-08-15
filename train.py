import os
from typing import List

import fire
import numpy as np
import torch
import torch.utils.data as data
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import TQDMProgressBar
from pydantic import TypeAdapter

from modules.transcriber import Transcriber, TranscriberConfig
from training.config import DatasetConfig
from training.dataset import Dataset, Metadata
from training.module import TranscriberModule

class MyProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, pl_module):
        items = super().get_metrics(trainer, pl_module)
        items["all_loss_mean"] = np.mean(pl_module.all_loss or float("nan"))
        items["epoch_loss_mean"] = np.mean(pl_module.epoch_loss or float("nan"))
        return items

def main(
    dataset_config: str = "./dataset.json",
    dataset_dir: str = "./maestro-v3.0.0-preprocessed",
    accelerator: str = "gpu",
    devices: str = "0,",
    max_train_epochs: int = 100,
    batch_size: int = 1,
    num_workers: int = 1,
):
    metadata_path = os.path.join(dataset_dir, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata: List[Metadata] = TypeAdapter(List[Metadata]).validate_json(f.read())

    with open(dataset_config, "r") as f:
        config: DatasetConfig = DatasetConfig.model_validate_json(f.read())

    filenames = [
        m.midi_filename.replace("/", "-") + ".pt"
        for m in metadata
        if m.split == "train"
    ]

    dataset = Dataset(
        filenames=filenames,
        features_dir=os.path.join(dataset_dir, "features"),
        labels_dir=os.path.join(dataset_dir, "labels"),
    )
    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
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
    transcriber = Transcriber(params)
    module = TranscriberModule(transcriber, torch.optim.Adam)
    callbacks = [MyProgressBar()]
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_train_epochs,
        log_every_n_steps=1,
        callbacks=callbacks,
        precision="bf16",
    )
    trainer.fit(module, dataloader)


if __name__ == "__main__":
    fire.Fire(main)

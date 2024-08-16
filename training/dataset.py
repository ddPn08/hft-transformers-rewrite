import os
from typing import Dict, List, Literal

import torch
import torch.utils.data as data
from pydantic import BaseModel, TypeAdapter

from training.config import DatasetConfig


class Metadata(BaseModel):
    canonical_composer: str
    canonical_title: str
    split: str
    year: int
    midi_filename: str
    audio_filename: str
    duration: float


class FrameInfomation(BaseModel):
    onset_frame: int
    offset_frame: int


class DatasetItem(BaseModel):
    split: Literal['train', 'validation', 'test']
    basename: str
    feature: FrameInfomation
    label: FrameInfomation


class Dataset(data.Dataset):
    def __init__(
        self,
        dir: str,
        split: str = "train",
        num_frames: int = 128,
    ):
        datamapping_path = os.path.join(dir, "mapping.json")
        with open(datamapping_path, "r") as f:
            self.datamapping = TypeAdapter(List[DatasetItem]).validate_json(f.read())
            self.datamapping = [item for item in self.datamapping if item.split == split]
        config_path = os.path.join(dir, "config.json")
        with open(config_path, "r") as f:
            self.config = DatasetConfig.model_validate_json(f.read())
        self.features_dir = os.path.join(dir, "features")
        self.labels_dir = os.path.join(dir, "labels")
        self.num_frames = num_frames

    def __getitem__(self, idx: int):
        mapping = self.datamapping[idx]

        feature_path = os.path.join(self.features_dir, mapping.split, mapping.basename + ".pt")
        label_path = os.path.join(self.labels_dir, mapping.split, mapping.basename + ".pt")

        feature: torch.Tensor = torch.load(
            feature_path, map_location="cpu", weights_only=True
        )
        labels: Dict[str, torch.Tensor] = torch.load(
            label_path, map_location="cpu", weights_only=True
        )

        zero_value = torch.log(torch.tensor(self.config.feature.log_offset))
        if mapping.feature.onset_frame < 0:
            pad = torch.zeros(
                -mapping.feature.onset_frame, feature.shape[1], dtype=feature.dtype
            )
            feature = torch.cat([pad.fill_(zero_value), feature], dim=0)
            mapping.feature.onset_frame = 0
            mapping.feature.offset_frame = (
                mapping.feature.offset_frame - mapping.feature.onset_frame
            )

        feature = feature[mapping.feature.onset_frame : mapping.feature.offset_frame]
        num_feature_frames = (
            self.config.input.margin_b + self.num_frames + self.config.input.margin_f
        )
        if feature.shape[0] < num_feature_frames:
            pad = torch.zeros(
                num_feature_frames - feature.shape[0],
                feature.shape[1],
                dtype=feature.dtype,
            )
            feature = torch.cat([feature, pad.fill_(zero_value)], dim=0)

        spec = feature.T

        for label in labels:
            tensor = labels[label][
                mapping.label.onset_frame : mapping.label.offset_frame
            ]
            if tensor.shape[0] < self.num_frames:
                pad = torch.zeros(
                    self.num_frames - tensor.shape[0],
                    tensor.shape[1],
                    dtype=tensor.dtype,
                )
                tensor = torch.cat([tensor, pad], dim=0)
            labels[label] = tensor

        onset = labels["onset"]
        offset = labels["offset"]
        onpedal = labels["onpedal"]
        offpedal = labels["offpedal"]
        mpe = labels["mpe"].float()
        mpe_pedal = labels["mpe_pedal"].float()
        velocity = labels["velocity"].long()

        return spec, onset, offset, onpedal, offpedal, mpe, mpe_pedal, velocity

    def __len__(self):
        return len(self.datamapping)

    def collate_fn(self, batch):
        specs, onsets, offsets, onpedals, offpedals, mpes, mpes_pedal, velocities = zip(*batch)
        specs = torch.stack(specs)
        onsets = torch.stack(onsets)
        offsets = torch.stack(offsets)
        onpedals = torch.stack(onpedals)
        offpedals = torch.stack(offpedals)
        mpes = torch.stack(mpes)
        mpes_pedal = torch.stack(mpes_pedal)
        velocities = torch.stack(velocities)
        return specs, onsets, offsets, onpedals, offpedals, mpes, mpes_pedal, velocities

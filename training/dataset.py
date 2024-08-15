import json
import os
from typing import List

import numpy as np
from pydantic import BaseModel
import torch
import torch.utils.data as data

from preprocess.midi import LABELS


class Metadata(BaseModel):
    canonical_composer: str
    canonical_title: str
    split: str
    year: int
    midi_filename: str
    audio_filename: str
    duration: float

class Dataset(data.Dataset):
    def __init__(
        self,
        filenames: List[str],
        features_dir: str,
        labels_dir: str,
        num_frames: int = 128,
    ):
        self.filenames = filenames
        self.features_dir = features_dir
        self.labels_dir = labels_dir
        self.num_frames = num_frames

    def __getitem__(self, idx: int):
        filename = self.filenames[idx]
        feature_path = os.path.join(self.features_dir, filename)
        labels = {}
        for label in LABELS:
            label_path = os.path.join(
                self.labels_dir, filename.replace(".pt", f".{label}.json")
            )
            with open(label_path, "r") as f:
                arr = json.load(f)
                labels[label] = torch.tensor(arr)

        feature: torch.Tensor = torch.load(feature_path, map_location="cpu", weights_only=True)

        start_frame = np.random.randint(0, feature.shape[0] - self.num_frames)
        end_frame = start_frame + self.num_frames

        spec_start_frame = start_frame - 32
        spec_end_frame = end_frame + 32

        spec = (feature[spec_start_frame:spec_end_frame]).T
        for label in labels:
            labels[label] = labels[label][start_frame:end_frame]

        onset = labels["onset"]
        offset = labels["offset"]
        mpe = labels["mpe"].float()
        velocity = labels["velocity"].long()

        return spec, onset, offset, mpe, velocity

    def __len__(self):
        return len(self.filenames)

    def collate_fn(self, batch):
        specs, onsets, offsets, mpes, velocities = zip(*batch)
        specs = torch.stack(specs)
        onsets = torch.stack(onsets)
        offsets = torch.stack(offsets)
        mpes = torch.stack(mpes)
        velocities = torch.stack(velocities)
        return specs, onsets, offsets, mpes, velocities

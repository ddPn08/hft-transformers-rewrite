import json
import os
from typing import List

import fire
import numpy as np
import torch
import torch.multiprocessing as mp
import torchaudio
import tqdm
from pydantic import RootModel

from preprocess.midi import LABELS, create_label, create_note
from training.config import DatasetConfig
from training.dataset import Metadata
from utils.logger import get_logger

logger = get_logger(__name__)


def process_metadata(
    idx: int,
    metadata: List[Metadata],
    dataset_path: str,
    config: DatasetConfig,
    label_dir: str,
    features_dir: str,
    force_reprocess: bool,
    device: str,
):
    for m in tqdm.tqdm(metadata, desc=f"CreateLabel {idx}", position=idx):
        basename = m.midi_filename.replace("/", "-")

        all_exists = all(
            os.path.exists(os.path.join(label_dir, f"{basename}.{label}.json"))
            for label in LABELS
        )
        if all_exists and not force_reprocess:
            continue

        notes = create_note(
            os.path.join(dataset_path, m.midi_filename),
            min_pitch=config.midi.pitch_min,
            max_pitch=config.midi.pitch_max,
            apply_pedal=True,
        )

        labels = create_label(config.feature, config.midi, notes)

        for label, data in labels.items():
            with open(os.path.join(label_dir, f"{basename}.{label}.json"), "w") as f:
                json.dump(data, f)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.feature.sampling_rate,
        n_fft=config.feature.fft_bins,
        win_length=config.feature.window_length,
        hop_length=config.feature.hop_sample,
        pad_mode=config.feature.pad_mode,
        n_mels=config.feature.mel_bins,
        norm="slaney",
    ).to(device)

    for m in tqdm.tqdm(metadata, desc=f"CreateLogMelSpec {idx}", position=idx):
        basename = m.midi_filename.replace("/", "-")
        log_melspec_path = os.path.join(features_dir, f"{basename}.pt")

        if os.path.exists(log_melspec_path) and not force_reprocess:
            continue

        wav, sr = torchaudio.load(os.path.join(dataset_path, m.audio_filename))
        if device is not None:
            wav = wav.to(device)
        wav = wav.mean(0)
        if sr != config.feature.sampling_rate:
            wav = torchaudio.functional.resample(wav, sr, config.feature.sampling_rate)

        melspec = mel_transform(wav)
        log_melspec = (torch.log(melspec + config.feature.log_offset)).T

        torch.save(
            log_melspec,
            log_melspec_path,
        )


def main(
    dataset_config: str = "dataset.json",
    dataset_path: str = "maestro-v3.0.0",
    dest_path: str = "maestro-v3.0.0-preprocessed",
    num_workers: int = 4,
    device: str = "cuda",
    force_reprocess: bool = False,
    max_value: float = 0.0,
):
    with open(dataset_config, "r") as f:
        config = DatasetConfig.model_validate(json.load(f))

    with open(os.path.join(dataset_path, "maestro-v3.0.0.json"), "r") as f:
        raw_metadata = json.load(f)

    metadata: List[Metadata] = []
    keys = list(raw_metadata.keys())

    for idx in range(len(raw_metadata[keys[0]])):
        data = {}
        for key in keys:
            data[key] = raw_metadata[key][str(idx)]
        metadata.append(Metadata.model_validate(data))

    metadata = [m for m in metadata if m.year == 2015 and m.split == "train"]

    metadata_path = os.path.join(dest_path, "metadata.json")
    label_dir = os.path.join(dest_path, "labels")
    features_dir = os.path.join(dest_path, "features")
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)

    with open(metadata_path, "w") as f:
        f.write(RootModel(metadata).model_dump_json())

    config.input.max_value = max_value

    if config.feature.log_offset > 0.0:
        config.input.min_value = np.log(config.feature.log_offset).astype(np.float32)
    else:
        config.input.min_value = config.feature.log_offset

    processes = []
    for idx in range(num_workers):
        p = mp.Process(
            target=process_metadata,
            args=(
                idx,
                metadata[idx::num_workers],
                dataset_path,
                config,
                label_dir,
                features_dir,
                force_reprocess,
                device,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    fire.Fire(main)

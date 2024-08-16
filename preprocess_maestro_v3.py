import json
import os
from concurrent.futures import ProcessPoolExecutor, Future
from typing import List

import fire
import numpy as np
import torch
import torch.multiprocessing as mp
import torchaudio
import tqdm
from pydantic import RootModel

from preprocess.midi import create_label, create_note
from training.config import DatasetConfig
from training.dataset import DatasetItem, FrameInfomation, Metadata
from utils.logger import get_logger

logger = get_logger(__name__)


def process_metadata(
    idx: int,
    metadata: List[Metadata],
    dataset_path: str,
    config: DatasetConfig,
    label_dir: str,
    force_reprocess: bool,
):
    for m in tqdm.tqdm(metadata, desc=f"CreateLabel {idx}", position=idx):
        basename = os.path.basename(m.midi_filename.replace("/", "-"))

        label_path = os.path.join(label_dir, m.split, f"{basename}.pt")
        os.makedirs(os.path.dirname(label_path), exist_ok=True)

        if os.path.exists(label_path) and not force_reprocess:
            continue

        try:
            notes = create_note(
                os.path.join(dataset_path, m.midi_filename),
                min_pitch=config.midi.pitch_min,
                max_pitch=config.midi.pitch_max,
                apply_pedal=True,
            )

            labels = create_label(config.feature, config.midi, notes)

            labels = {k: torch.tensor(v) for k, v in labels.items()}

            torch.save(labels, label_path)
        except Exception as e:
            logger.error(f"Error: {basename}")
            logger.error(e)
            raise e


def process_melspec(
    idx: int,
    metadata: List[Metadata],
    dataset_path: str,
    config: DatasetConfig,
    features_dir: str,
    force_reprocess: bool,
    device: str,
):
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
        basename = os.path.basename(m.midi_filename.replace("/", "-"))
        log_melspec_path = os.path.join(features_dir, m.split, f"{basename}.pt")

        if os.path.exists(log_melspec_path) and not force_reprocess:
            continue

        os.makedirs(os.path.dirname(log_melspec_path), exist_ok=True)

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


def mapping_dataset(
    metadata: List[Metadata],
    dataset_path: str,
    config: DatasetConfig,
):
    dataset: List[DatasetItem] = []
    for m in tqdm.tqdm(metadata):
        basename = os.path.basename(m.midi_filename.replace("/", "-"))
        feature = torch.load(
            os.path.join(dataset_path, "features", m.split, f"{basename}.pt"),
            weights_only=True,
        )
        labels = torch.load(
            os.path.join(dataset_path, "labels", m.split, f"{basename}.pt"),
            weights_only=True,
        )

        num_frames = feature.shape[0]
        label_num_frames = labels["mpe"].shape[0]

        if num_frames < label_num_frames:
            logger.warning(f"Feature frames are less than label frames: {basename}")

        num_frames = max(num_frames, label_num_frames)

        items = num_frames // config.input.num_frame

        for j in range(items):
            start_frame = j * config.input.num_frame
            end_frame = start_frame + config.input.num_frame

            spec_start_frame = start_frame - config.input.margin_b
            spec_end_frame = end_frame + config.input.margin_f

            if spec_end_frame > num_frames:
                continue

            dataset.append(
                DatasetItem(
                    basename=basename,
                    split=m.split,
                    feature=FrameInfomation(
                        onset_frame=spec_start_frame, offset_frame=spec_end_frame
                    ),
                    label=FrameInfomation(
                        onset_frame=start_frame, offset_frame=end_frame
                    ),
                )
            )

    return dataset


def main(
    dataset_config: str = "dataset.json",
    dataset_path: str = "maestro-v3.0.0",
    dest_path: str = "maestro-v3.0.0-preprocessed",
    num_workers: int = 4,
    num_gpu_workers: int = 1,
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

    metadata = [m for m in metadata if m.year == 2009]

    label_dir = os.path.join(dest_path, "labels")
    features_dir = os.path.join(dest_path, "features")
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)

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
                force_reprocess,
            ),
        )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

    processes = []
    for idx in range(num_gpu_workers):
        p = mp.Process(
            target=process_melspec,
            args=(
                idx,
                metadata[idx::num_gpu_workers],
                dataset_path,
                config,
                features_dir,
                force_reprocess,
                device,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    mapping = mapping_dataset(metadata, dest_path, config)

    mapping_path = os.path.join(dest_path, "mapping.json")
    with open(mapping_path, "w") as f:
        f.write(RootModel(mapping).model_dump_json())

    config_path = os.path.join(dest_path, "config.json")
    with open(config_path, "w") as f:
        f.write(config.model_dump_json())


if __name__ == "__main__":
    fire.Fire(main)

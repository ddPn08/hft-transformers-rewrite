import torch
import torchaudio

from training.config import FeatureConfig


def create_log_melspec(filepath: str, config: FeatureConfig, device: torch.device = None) -> torch.Tensor:
    wav, sr = torchaudio.load(filepath)
    if device is not None:
        wav = wav.to(device)
    wav = wav.mean(0)
    if sr != config.sampling_rate:
        wav = torchaudio.functional.resample(wav, sr, config.sampling_rate)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.sampling_rate,
        n_fft=config.fft_bins,
        win_length=config.window_length,
        hop_length=config.hop_sample,
        pad_mode=config.pad_mode,
        n_mels=config.mel_bins,
        norm="slaney",
    ).to(device)

    melspec = mel_transform(wav)
    log_melspec = (torch.log(melspec + config.log_offset)).T

    return log_melspec

from typing import Literal, Optional
from pydantic import BaseModel

from modules.transcriber import TranscriberConfig


class FeatureConfig(BaseModel):
    sampling_rate: int = 16000
    hop_sample: int = 256
    mel_bins: int = 256
    n_bins: int = 256
    fft_bins: int = 2048
    window_length: int = 2048
    log_offset: float = 1e-8
    window_mode: str = "hann"
    pad_mode: str = "constant"


class MidiConfig(BaseModel):
    pitch_min: int = 21
    pitch_max: int = 108
    num_notes: int = 88
    num_velocity: int = 128


class InputConfig(BaseModel):
    margin_b: int = 32
    margin_f: int = 32
    num_frame: int = 128
    max_value: Optional[float] = None
    min_value: Optional[float] = None


class DatasetConfig(BaseModel):
    feature: FeatureConfig
    input: InputConfig
    midi: MidiConfig

class ModelConfig(BaseModel):
    mode: Literal["note", "pedal"]
    params: TranscriberConfig
    feature: FeatureConfig
    input: InputConfig
    midi: MidiConfig
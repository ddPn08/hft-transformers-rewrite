import fire
import numpy as np
import pretty_midi as pm
import torch
import torchaudio
import tqdm

from modules.transcriber import Transcriber
from preprocess.midi import convert_label_to_note
from training.config import ModelConfig


def main(
    wav_path: str,
    output_path: str,
    device="cuda",
    model_path: str = "model.pt",
    pedal_model_path: str = None,
    config_path: str = "config.json",
    thred_onset: float = 0.5,
    thred_offset: float = 0.5,
    thred_onpedal: float = 0.5,
    thred_offpedal: float = 0.5,
    thred_mpe: float = 0.5,
    thred_mpe_pedal: float = 0.5,
):
    device = torch.device(device)
    with open(config_path, "r") as f:
        config = ModelConfig.model_validate_json(f.read())

    wav, sr = torchaudio.load(wav_path)
    if device is not None:
        wav = wav.to(device)
    wav = wav.mean(0)
    if sr != config.feature.sampling_rate:
        wav = torchaudio.functional.resample(wav, sr, config.feature.sampling_rate)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.feature.sampling_rate,
        n_fft=config.feature.fft_bins,
        win_length=config.feature.window_length,
        hop_length=config.feature.hop_sample,
        pad_mode=config.feature.pad_mode,
        n_mels=config.feature.mel_bins,
        norm="slaney",
    ).to(device)

    melspec = mel_transform(wav)
    feature = (torch.log(melspec + config.feature.log_offset)).T

    # a_tmp_b = np.full([self.config['input']['margin_b'], self.config['feature']['n_bins']], self.config['input']['min_value'], dtype=np.float32)
    # len_s = int(np.ceil(a_feature.shape[0] / self.config['input']['num_frame']) * self.config['input']['num_frame']) - a_feature.shape[0]
    # a_tmp_f = np.full([len_s+self.config['input']['margin_f'], self.config['feature']['n_bins']], self.config['input']['min_value'], dtype=np.float32)
    # a_input = torch.from_numpy(np.concatenate([a_tmp_b, a_feature, a_tmp_f], axis=0))
    a_tmp_b = torch.full(
        [config.input.margin_b, config.feature.n_bins],
        config.input.min_value,
        dtype=torch.float32,
        device=device,
    )
    len_s = (
        int(np.ceil(feature.shape[0] / config.input.num_frame) * config.input.num_frame)
        - feature.shape[0]
    )
    a_tmp_f = torch.full(
        [len_s + config.input.margin_f, config.feature.n_bins],
        config.input.min_value,
        dtype=torch.float32,
        device=device,
    )
    feature_with_margin = torch.cat([a_tmp_b, feature, a_tmp_f], axis=0)

    state_dict = torch.load(model_path, map_location=device)

    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if any([p.startswith("model.") for p in state_dict.keys()]):
        state_dict = {k[6:]: v for k, v in state_dict.items() if k.startswith("model.")}

    model = Transcriber(config.params, mode="note").to(device)
    model.load_state_dict(state_dict)
    model.eval()

    pedal_model = None
    if pedal_model_path is not None:
        state_dict_pedal = torch.load(pedal_model_path, map_location=device)

        if "state_dict" in state_dict_pedal:
            state_dict_pedal = state_dict_pedal["state_dict"]

        if any([p.startswith("model.") for p in state_dict_pedal.keys()]):
            state_dict_pedal = {
                k[6:]: v for k, v in state_dict_pedal.items() if k.startswith("model.")
            }

        pedal_model = Transcriber(config.params, mode="pedal").to(device)
        pedal_model.load_state_dict(state_dict_pedal)
        pedal_model.eval()

    output_onset_A_all = np.zeros(
        (feature.shape[0] + len_s, config.midi.num_notes), dtype=np.float32
    )
    output_offset_A_all = np.zeros(
        (feature.shape[0] + len_s, config.midi.num_notes), dtype=np.float32
    )
    output_onpedal_A_all = np.zeros(feature.shape[0] + len_s, dtype=np.float32)
    output_offpedal_A_all = np.zeros(feature.shape[0] + len_s, dtype=np.float32)
    output_mpe_A_all = np.zeros(
        (feature.shape[0] + len_s, config.midi.num_notes), dtype=np.float32
    )
    output_mpe_pedal_A_all = np.zeros(feature.shape[0] + len_s, dtype=np.float32)
    output_velocity_A_all = np.zeros(
        (feature.shape[0] + len_s, config.midi.num_notes), dtype=np.int8
    )

    output_onset_B_all = np.zeros(
        (feature.shape[0] + len_s, config.midi.num_notes), dtype=np.float32
    )
    output_offset_B_all = np.zeros(
        (feature.shape[0] + len_s, config.midi.num_notes), dtype=np.float32
    )
    output_onpedal_B_all = np.zeros(feature.shape[0] + len_s, dtype=np.float32)
    output_offpedal_B_all = np.zeros(feature.shape[0] + len_s, dtype=np.float32)
    output_mpe_B_all = np.zeros(
        (feature.shape[0] + len_s, config.midi.num_notes), dtype=np.float32
    )
    output_mpe_pedal_B_all = np.zeros(feature.shape[0] + len_s, dtype=np.float32)
    output_velocity_B_all = np.zeros(
        (feature.shape[0] + len_s, config.midi.num_notes), dtype=np.int8
    )

    for i in tqdm.tqdm(range(0, feature.shape[0], config.input.num_frame)):
        input = (
            (
                feature_with_margin[
                    i : i
                    + config.input.margin_b
                    + config.input.num_frame
                    + config.input.margin_f
                ]
            )
            .T.unsqueeze(0)
            .to(device)
        )
        with torch.no_grad():
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
            ) = model(input)
            output_onset_A = torch.sigmoid(output_onset_A)
            output_offset_A = torch.sigmoid(output_offset_A)
            output_mpe_A = torch.sigmoid(output_mpe_A)
            output_onset_B = torch.sigmoid(output_onset_B)
            output_offset_B = torch.sigmoid(output_offset_B)
            output_mpe_B = torch.sigmoid(output_mpe_B)

            if pedal_model is not None:
                (
                    output_onpedal_A,
                    output_offpedal_A,
                    output_mpe_pedal_A,
                    _,
                    output_onpedal_B,
                    output_offpedal_B,
                    output_mpe_pedal_B,
                ) = pedal_model(input)
                output_onpedal_A = torch.sigmoid(output_onpedal_A)
                output_offpedal_A = torch.sigmoid(output_offpedal_A)
                output_mpe_pedal_A = torch.sigmoid(output_mpe_pedal_A)
                output_onpedal_B = torch.sigmoid(output_onpedal_B)
                output_offpedal_B = torch.sigmoid(output_offpedal_B)
                output_mpe_pedal_B = torch.sigmoid(output_mpe_pedal_B)

        output_onset_A_all[i : i + config.input.num_frame] = (
            output_onset_A.squeeze(0).detach().to("cpu").numpy()
        )
        output_offset_A_all[i : i + config.input.num_frame] = (
            output_offset_A.squeeze(0).detach().to("cpu").numpy()
        )
        output_mpe_A_all[i : i + config.input.num_frame] = (
            output_mpe_A.squeeze(0).detach().to("cpu").numpy()
        )
        output_velocity_A_all[i : i + config.input.num_frame] = (
            output_velocity_A.squeeze(0).argmax(2).detach().to("cpu").numpy()
        )

        output_onset_B_all[i : i + config.input.num_frame] = (
            output_onset_B.squeeze(0).detach().to("cpu").numpy()
        )
        output_offset_B_all[i : i + config.input.num_frame] = (
            output_offset_B.squeeze(0).detach().to("cpu").numpy()
        )
        output_mpe_B_all[i : i + config.input.num_frame] = (
            output_mpe_B.squeeze(0).detach().to("cpu").numpy()
        )
        output_velocity_B_all[i : i + config.input.num_frame] = (
            output_velocity_B.squeeze(0).argmax(2).detach().to("cpu").numpy()
        )

        if pedal_model is not None:
            output_onpedal_A_all[i : i + config.input.num_frame] = (
                output_onpedal_A.squeeze(0).detach().to("cpu").numpy()
            )
            output_offpedal_A_all[i : i + config.input.num_frame] = (
                output_offpedal_A.squeeze(0).detach().to("cpu").numpy()
            )
            output_onpedal_B_all[i : i + config.input.num_frame] = (
                output_onpedal_B.squeeze(0).detach().to("cpu").numpy()
            )
            output_offpedal_B_all[i : i + config.input.num_frame] = (
                output_offpedal_B.squeeze(0).detach().to("cpu").numpy()
            )
            output_mpe_pedal_A_all[i : i + config.input.num_frame] = (
                output_mpe_pedal_A.squeeze(0).detach().to("cpu").numpy()
            )
            output_mpe_pedal_B_all[i : i + config.input.num_frame] = (
                output_mpe_pedal_B.squeeze(0).detach().to("cpu").numpy()
            )

    notes_A, pedals_A = convert_label_to_note(
        config.feature,
        config.midi,
        output_onset_A_all,
        output_offset_A_all,
        output_onpedal_A_all,
        output_offpedal_A_all,
        output_mpe_A_all,
        output_mpe_pedal_A_all,
        output_velocity_A_all,
        thred_onset=thred_onset,
        thred_offset=thred_offset,
        thred_onpedal=thred_onpedal,
        thred_offpedal=thred_offpedal,
        thred_mpe=thred_mpe,
        thred_mpe_pedal=thred_mpe_pedal,
        mode_velocity="ignore_zero",
        mode_offset="shorter",
    )

    note_B, pedals_B = convert_label_to_note(
        config.feature,
        config.midi,
        output_onset_B_all,
        output_offset_B_all,
        output_onpedal_B_all,
        output_offpedal_B_all,
        output_mpe_B_all,
        output_mpe_pedal_B_all,
        output_velocity_B_all,
        thred_onset=thred_onset,
        thred_offset=thred_offset,
        thred_onpedal=thred_onpedal,
        thred_offpedal=thred_offpedal,
        thred_mpe=thred_mpe,
        thred_mpe_pedal=thred_mpe_pedal,
        mode_velocity="ignore_zero",
        mode_offset="shorter",
    )

    notes_A.extend(note_B)
    pedals_A.extend(pedals_B)

    if len(notes_A) == 0:
        raise ValueError("No notes detected.")

    midi = pm.PrettyMIDI()
    instrument = pm.Instrument(program=0)

    for pedal in pedals_A:
        instrument.control_changes.append(
            pm.ControlChange(number=64, value=127, time=pedal.onset)
        )

        instrument.control_changes.append(
            pm.ControlChange(number=64, value=0, time=pedal.offset)
        )

    for note in notes_A:
        instrument.notes.append(
            pm.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=note.onset,
                end=note.offset,
            )
        )

    midi.instruments.append(instrument)
    midi.write(output_path)


if __name__ == "__main__":
    fire.Fire(main)

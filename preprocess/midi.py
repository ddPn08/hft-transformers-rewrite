from typing import Dict, List, Union

import numpy as np
import pretty_midi as pm
from pydantic import BaseModel

from training.config import FeatureConfig, MidiConfig
from utils.logger import get_logger

logger = get_logger(__name__)

LABELS = [
    "onset",
    "offset",
    "mpe",
    "velocity",
]


class PedalOn(BaseModel):
    time: float


class PedalOff(BaseModel):
    time: float


class NoteOn(BaseModel):
    time: float
    pitch: int
    velocity: int


class NoteOff(BaseModel):
    time: float
    pitch: int


class NoteState(BaseModel):
    onset: float = 0
    offset: float = 0
    pitch: int = 0
    velocity: int = 0
    on: bool = False
    reonset: bool = False
    sustain: bool = False


class Note(BaseModel):
    onset: float
    offset: float
    pitch: int
    velocity: int
    reonset: bool

    def from_state(state: NoteState):
        return Note(
            onset=state.onset,
            offset=state.offset,
            pitch=state.pitch,
            velocity=state.velocity,
            reonset=state.reonset,
        )


def create_note(
    filepath: str, min_pitch: int = 21, max_pitch: int = 108, apply_pedal: bool = True
):
    midi = pm.PrettyMIDI(filepath)

    events: List[Union[PedalOn, PedalOff, NoteOn, NoteOff]] = []

    if apply_pedal:
        for cc in midi.instruments[0].control_changes:
            if cc.number != 64:
                continue
            if cc.value > 64:
                events.append(PedalOn(time=cc.time))
            else:
                events.append(PedalOff(time=cc.time))

    for note in midi.instruments[0].notes:
        if note.pitch < min_pitch or note.pitch > max_pitch:
            continue
        events.append(NoteOn(time=note.start, pitch=note.pitch, velocity=note.velocity))
        events.append(NoteOff(time=note.end, pitch=note.pitch))

    events.sort(key=lambda x: x.time)

    sustain = False
    note_states: Dict[int, NoteState] = {}

    notes = []

    for event in events:
        if isinstance(event, PedalOn):
            sustain = True
            for state in note_states.values():
                if state is None:
                    continue
                state.sustain = True
        elif isinstance(event, PedalOff):
            for state in note_states.values():
                if state is None:
                    continue
                if not state.on and state.sustain:
                    state.offset = event.time
                    notes.append(Note.from_state(state))
                    note_states[state.pitch] = None

            sustain = False
        elif isinstance(event, NoteOn):
            note = note_states[event.pitch] if event.pitch in note_states else None
            reonset = False
            if note is not None and note.sustain:
                note.offset = event.time
                notes.append(Note.from_state(note))
                reonset = True

            note = NoteState(
                onset=event.time,
                pitch=event.pitch,
                velocity=event.velocity,
                sustain=sustain,
                reonset=reonset,
                on=True,
            )
            note_states[event.pitch] = note

        elif isinstance(event, NoteOff):
            note = note_states[event.pitch] if event.pitch in note_states else None
            if note is None:
                logger.warning(f"NoteOff event without NoteOn: {event}")
                continue

            if note.sustain:
                note.on = False
                continue

            note.offset = event.time
            notes.append(Note.from_state(note))
            note_states[event.pitch] = None

    notes = sorted(sorted(notes, key=lambda x: x.pitch), key=lambda x: x.onset)

    return notes


def create_label(
    feature_config: FeatureConfig,
    midi_config: MidiConfig,
    notes: List[Note],
    offset_duration_tolerance_flag: bool = False,
):
    hop_ms = 1000 * feature_config.hop_sample / feature_config.sampling_rate
    onset_tolerance = int(50.0 / hop_ms + 0.5)
    offset_tolerance = int(50.0 / hop_ms + 0.5)

    num_frame_in_sec = feature_config.sampling_rate / feature_config.hop_sample

    max_offset = max([note.offset for note in notes])

    num_frame = int(max_offset * num_frame_in_sec + 0.5) + 1

    a_mpe = np.zeros((num_frame, midi_config.num_notes), dtype=np.bool)
    a_onset = np.zeros((num_frame, midi_config.num_notes), dtype=np.float32)
    a_offset = np.zeros((num_frame, midi_config.num_notes), dtype=np.float32)
    a_velocity = np.zeros((num_frame, midi_config.num_notes), dtype=np.int8)

    for note in notes:
        pitch = note.pitch - midi_config.pitch_min

        onset_frame = int(note.onset * num_frame_in_sec + 0.5)
        onset_ms = note.onset * 1000.0
        onset_sharpness = onset_tolerance

        offset_frame = int(note.offset * num_frame_in_sec + 0.5)
        offset_ms = note.offset * 1000.0
        offset_sharpness = offset_tolerance

        if offset_duration_tolerance_flag:
            offset_duration_tolerance = int((offset_ms - onset_ms) * 0.2 / hop_ms + 0.5)
            offset_sharpness = max(offset_tolerance, offset_duration_tolerance)

        velocity = note.velocity

        for j in range(0, onset_sharpness + 1):
            onset_ms_q = (onset_frame + j) * hop_ms
            onset_ms_diff = onset_ms_q - onset_ms
            onset_val = max(
                0.0, 1.0 - (abs(onset_ms_diff) / (onset_sharpness * hop_ms))
            )
            if onset_frame + j < num_frame:
                a_onset[onset_frame + j][pitch] = max(
                    a_onset[onset_frame + j][pitch], onset_val
                )
                if a_onset[onset_frame + j][pitch] >= 0.5:
                    a_velocity[onset_frame + j][pitch] = velocity

        for j in range(1, onset_sharpness + 1):
            onset_ms_q = (onset_frame - j) * hop_ms
            onset_ms_diff = onset_ms_q - onset_ms
            onset_val = max(
                0.0, 1.0 - (abs(onset_ms_diff) / (onset_sharpness * hop_ms))
            )
            if onset_frame - j >= 0:
                a_onset[onset_frame - j][pitch] = max(
                    a_onset[onset_frame - j][pitch], onset_val
                )
                if (a_onset[onset_frame - j][pitch] >= 0.5) and (
                    a_velocity[onset_frame - j][pitch] == 0
                ):
                    a_velocity[onset_frame - j][pitch] = velocity

        # mpe
        for j in range(onset_frame, offset_frame + 1):
            a_mpe[j][pitch] = 1

        # offset
        offset_flag = True
        for j in range(len(notes)):
            note_2 = notes[j]
            if note.pitch != note_2.pitch:
                continue
            if note.offset == note_2.offset:
                offset_flag = False
                break

        if offset_flag is True:
            for j in range(0, offset_sharpness + 1):
                offset_ms_q = (offset_frame + j) * hop_ms
                offset_ms_diff = offset_ms_q - offset_ms
                offset_val = max(
                    0.0, 1.0 - (abs(offset_ms_diff) / (offset_sharpness * hop_ms))
                )
                if offset_frame + j < num_frame:
                    a_offset[offset_frame + j][pitch] = max(
                        a_offset[offset_frame + j][pitch], offset_val
                    )
            for j in range(1, offset_sharpness + 1):
                offset_ms_q = (offset_frame - j) * hop_ms
                offset_ms_diff = offset_ms_q - offset_ms
                offset_val = max(
                    0.0, 1.0 - (abs(offset_ms_diff) / (offset_sharpness * hop_ms))
                )
                if offset_frame - j >= 0:
                    a_offset[offset_frame - j][pitch] = max(
                        a_offset[offset_frame - j][pitch], offset_val
                    )

    return {
        "onset": a_onset.tolist(),
        "offset": a_offset.tolist(),
        "mpe": a_mpe.tolist(),
        "velocity": a_velocity.tolist(),
    }

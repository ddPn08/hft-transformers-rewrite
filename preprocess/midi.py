from typing import Dict, List, Tuple, Union

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
    onset: float = -1.0
    offset: float = -1.0
    pitch: int = 0
    velocity: int = 0
    reonset: bool = False


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


class PedalWithPitch(BaseModel):
    onset: float
    offset: float
    pitch: int = 0


class PedalState(BaseModel):
    onset: float = -1.0
    offset: float = -1.0


class Pedal(BaseModel):
    onset: float
    offset: float

    def from_state(state: PedalState):
        return Pedal(onset=state.onset, offset=state.offset)


def create_note(
    filepath: str, min_pitch: int = 21, max_pitch: int = 108, apply_pedal: bool = True
) -> Tuple[List[Note], List[Pedal]]:
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
    pedal_state: PedalState = PedalState()

    notes: List[Note] = []
    pedals: List[Pedal] = []

    for event in events:
        if isinstance(event, PedalOn):
            sustain = True
            if pedal_state.onset < 0:
                pedal_state.onset = event.time

        elif isinstance(event, PedalOff):
            sustain = False
            if pedal_state.onset < 0 and len(pedals) > 0:
                pedals[-1].offset = event.time
            elif pedal_state.offset < 0:
                pedal_state.offset = event.time
                pedals.append(Pedal.from_state(pedal_state))
                pedal_state = PedalState()

        elif isinstance(event, NoteOn):
            state = note_states[event.pitch] if event.pitch in note_states else None
            reonset = False
            if state is not None:
                state.offset = event.time
                notes.append(Note.from_state(state))
                reonset = True

            state = NoteState(
                onset=event.time,
                pitch=event.pitch,
                velocity=event.velocity,
                sustain=sustain,
                reonset=reonset,
            )
            note_states[event.pitch] = state

        elif isinstance(event, NoteOff):
            state = note_states[event.pitch] if event.pitch in note_states else None
            if state is None:
                logger.warning(f"NoteOff event without NoteOn: {event}")
                continue

            state.offset = event.time

            notes.append(Note.from_state(state))
            note_states[event.pitch] = None

    notes = sorted(sorted(notes, key=lambda x: x.pitch), key=lambda x: x.onset)

    return notes, pedals


def create_label(
    feature_config: FeatureConfig,
    midi_config: MidiConfig,
    notes: List[Note],
    pedals: List[Pedal],
    offset_duration_tolerance_flag: bool = False,
):
    hop_ms = 1000 * feature_config.hop_sample / feature_config.sampling_rate
    onset_tolerance = int(50.0 / hop_ms + 0.5)
    offset_tolerance = int(50.0 / hop_ms + 0.5)

    num_frame_in_sec = feature_config.sampling_rate / feature_config.hop_sample

    max_offset = max(
        max([note.offset for note in notes]),
        max([pedal.offset for pedal in pedals]) if len(pedals) > 0 else 0,
    )

    num_frame = int(max_offset * num_frame_in_sec + 0.5) + 1

    a_mpe = np.zeros((num_frame, midi_config.num_notes), dtype=np.bool_)
    a_mpe_pedal = np.zeros(num_frame, dtype=np.bool_)
    a_onset = np.zeros((num_frame, midi_config.num_notes), dtype=np.float32)
    a_offset = np.zeros((num_frame, midi_config.num_notes), dtype=np.float32)
    a_onpedal = np.zeros(num_frame, dtype=np.float32)
    a_offpedal = np.zeros(num_frame, dtype=np.float32)
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

        # offset
        offset_flag = True
        for j in range(len(notes)):
            note_2 = notes[j]
            if note.pitch != note_2.pitch:
                continue
            if note.offset == note_2.onset:
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

        # mpe
        a_mpe[onset_frame : offset_frame + 1, pitch] = 1

    for pedal in pedals:
        onpedal_frame = int(pedal.onset * num_frame_in_sec + 0.5)
        onpedal_ms = pedal.onset * 1000.0
        onpedal_sharpness = onset_tolerance

        offpedal_frame = int(pedal.offset * num_frame_in_sec + 0.5)
        offpedal_ms = pedal.offset * 1000.0
        offpedal_sharpness = offset_tolerance

        if offset_duration_tolerance_flag:
            offpedal_duration_tolerance = int(
                (offpedal_ms - onpedal_ms) * 0.2 / hop_ms + 0.5
            )
            offpedal_sharpness = max(offset_tolerance, offpedal_duration_tolerance)

        # offpedal
        offpedal_flag = True
        for pedal_2 in pedals:
            if pedal.offset == pedal_2.onset:
                offpedal_flag = False
                break

        if offpedal_flag is True:
            for j in range(0, offpedal_sharpness + 1):
                offpedal_ms_q = (offpedal_frame + j) * hop_ms
                offpedal_ms_diff = offpedal_ms_q - offpedal_ms
                offpedal_val = max(
                    0.0, 1.0 - (abs(offpedal_ms_diff) / (offpedal_sharpness * hop_ms))
                )
                if offpedal_frame + j < num_frame:
                    a_offpedal[offpedal_frame + j] = max(
                        a_offpedal[offpedal_frame + j], offpedal_val
                    )
            for j in range(1, offpedal_sharpness + 1):
                offpedal_ms_q = (offpedal_frame - j) * hop_ms
                offpedal_ms_diff = offpedal_ms_q - offpedal_ms
                offpedal_val = max(
                    0.0, 1.0 - (abs(offpedal_ms_diff) / (offpedal_sharpness * hop_ms))
                )
                if offpedal_frame - j >= 0:
                    a_offpedal[offpedal_frame - j] = max(
                        a_offpedal[offpedal_frame - j], offpedal_val
                    )

        # onpedal
        for j in range(0, onpedal_sharpness + 1):
            onpedal_ms_q = (onpedal_frame + j) * hop_ms
            onpedal_ms_diff = onpedal_ms_q - onpedal_ms
            onpedal_val = max(
                0.0, 1.0 - (abs(onpedal_ms_diff) / (onpedal_sharpness * hop_ms))
            )
            if onpedal_frame + j < num_frame:
                a_onpedal[onpedal_frame + j] = max(
                    a_onpedal[onpedal_frame + j], onpedal_val
                )

        for j in range(1, onpedal_sharpness + 1):
            onpedal_ms_q = (onpedal_frame - j) * hop_ms
            onpedal_ms_diff = onpedal_ms_q - onpedal_ms
            onpedal_val = max(
                0.0, 1.0 - (abs(onpedal_ms_diff) / (onpedal_sharpness * hop_ms))
            )
            if onpedal_frame - j >= 0:
                a_onpedal[onpedal_frame - j] = max(
                    a_onpedal[onpedal_frame - j], onpedal_val
                )

        # pedal mpe
        a_mpe_pedal[onpedal_frame : offpedal_frame + 1] = 1

    return {
        "mpe": a_mpe.tolist(),
        "mpe_pedal": a_mpe_pedal.tolist(),
        "onset": a_onset.tolist(),
        "offset": a_offset.tolist(),
        "onpedal": a_onpedal.tolist(),
        "offpedal": a_offpedal.tolist(),
        "velocity": a_velocity.tolist(),
    }


class Detection(BaseModel):
    loc: int
    time: float


def detect_event(
    hop_sec: float,
    data: np.ndarray,
    pitch: int,
    thredhold: float,
):
    result: List[Detection] = []
    for i in range(len(data)):
        if data[i][pitch] >= thredhold:
            left_flag = True
            for ii in range(i - 1, -1, -1):
                if data[i][pitch] > data[ii][pitch]:
                    left_flag = True
                    break
                elif data[i][pitch] < data[ii][pitch]:
                    left_flag = False
                    break
            right_flag = True
            for ii in range(i + 1, len(data)):
                if data[i][pitch] > data[ii][pitch]:
                    right_flag = True
                    break
                elif data[i][pitch] < data[ii][pitch]:
                    right_flag = False
                    break

            if (left_flag is True) and (right_flag is True):
                if (i == 0) or (i == len(data) - 1):
                    time = i * hop_sec
                else:
                    if data[i - 1][pitch] == data[i + 1][pitch]:
                        time = i * hop_sec
                    elif data[i - 1][pitch] > data[i + 1][pitch]:
                        time = i * hop_sec - (
                            hop_sec
                            * 0.5
                            * (data[i - 1][pitch] - data[i + 1][pitch])
                            / (data[i][pitch] - data[i + 1][pitch])
                        )
                    else:
                        time = i * hop_sec + (
                            hop_sec
                            * 0.5
                            * (data[i + 1][pitch] - data[i - 1][pitch])
                            / (data[i][pitch] - data[i - 1][pitch])
                        )
                result.append(Detection(loc=i, time=time))

    return result


def process_label(
    midi_config: MidiConfig,
    hop_sec: float,
    pitch: int,
    onset_detections: List[Detection],
    offset_detections: List[Detection],
    mpe: np.ndarray,
    thred_mpe: float,
    velocity: np.ndarray = None,
    mode_offset="shorter",
):
    time_next = 0.0
    time_offset = 0.0
    time_mpe = 0.0
    for idx_on in range(len(onset_detections)):
        # onset
        loc_onset = onset_detections[idx_on].loc
        time_onset = onset_detections[idx_on].time

        if idx_on + 1 < len(onset_detections):
            loc_next = onset_detections[idx_on + 1].loc
            # time_next = loc_next * hop_sec
            time_next = onset_detections[idx_on + 1].time
        else:
            loc_next = len(mpe)
            time_next = (loc_next - 1) * hop_sec

        # offset
        loc_offset = loc_onset + 1
        flag_offset = False
        # time_offset = 0###
        for idx_off in range(len(offset_detections)):
            if loc_onset < offset_detections[idx_off].loc:
                loc_offset = offset_detections[idx_off].loc
                time_offset = offset_detections[idx_off].time
                flag_offset = True
                break
        if loc_offset > loc_next:
            loc_offset = loc_next
            time_offset = time_next

        # offset by MPE
        # (1frame longer)
        loc_mpe = loc_onset + 1
        flag_mpe = False
        # time_mpe = 0###
        for ii_mpe in range(loc_onset + 1, loc_next):
            if mpe[ii_mpe][pitch] < thred_mpe:
                loc_mpe = ii_mpe
                flag_mpe = True
                time_mpe = loc_mpe * hop_sec
                break
        """
        # (right algorighm)
        loc_mpe = loc_onset
        flag_mpe = False
        for ii_mpe in range(loc_onset+1, loc_next+1):
            if a_mpe[ii_mpe][j] < thred_mpe:
                loc_mpe = ii_mpe-1
                flag_mpe = True
                time_mpe = loc_mpe * hop_sec
                break
        """
        pitch_value = int(pitch + midi_config.pitch_min)
        velocity_value = int(velocity[loc_onset][pitch]) if velocity is not None else 0

        if (flag_offset is False) and (flag_mpe is False):
            offset_value = float(time_next)
        elif (flag_offset is True) and (flag_mpe is False):
            offset_value = float(time_offset)
        elif (flag_offset is False) and (flag_mpe is True):
            offset_value = float(time_mpe)
        else:
            if mode_offset == "offset":
                ## (a) offset
                offset_value = float(time_offset)
            elif mode_offset == "longer":
                ## (b) longer
                if loc_offset >= loc_mpe:
                    offset_value = float(time_offset)
                else:
                    offset_value = float(time_mpe)
            else:
                ## (c) shorter
                if loc_offset <= loc_mpe:
                    offset_value = float(time_offset)
                else:
                    offset_value = float(time_mpe)

        yield (time_onset, offset_value, pitch_value, velocity_value)


def convert_label_to_note(
    feature_config: FeatureConfig,
    midi_config: MidiConfig,
    onset: np.ndarray,
    offset: np.ndarray,
    onpedal: np.ndarray,
    offpedal: np.ndarray,
    mpe: np.ndarray,
    mpe_pedal: np.ndarray,
    velocity: np.ndarray,
    thred_onset=0.5,
    thred_offset=0.5,
    thred_onpedal=0.5,
    thred_offpedal=0.5,
    thred_mpe=0.5,
    thred_mpe_pedal=0.5,
    mode_velocity="ignore_zero",
    mode_offset="shorter",
):
    notes: List[Note] = []
    hop_sec = float(feature_config.hop_sample / feature_config.sampling_rate)

    for pitch in range(midi_config.num_notes):
        # find local maximum
        a_onset_detect = detect_event(hop_sec, onset, pitch, thred_onset)
        a_offset_detect = detect_event(hop_sec, offset, pitch, thred_offset)

        for time_onset, offset_value, pitch_value, velocity_value in process_label(
            midi_config,
            hop_sec,
            pitch,
            a_onset_detect,
            a_offset_detect,
            mpe,
            thred_mpe,
            velocity,
            mode_offset,
        ):
            if mode_velocity != "ignore_zero":
                notes.append(
                    Note(
                        onset=float(time_onset),
                        offset=offset_value,
                        onpedal=0.0,
                        offpedal=0.0,
                        pitch=pitch_value,
                        velocity=velocity_value,
                        reonset=False,
                    )
                )
            else:
                if velocity_value > 0:
                    notes.append(
                        Note(
                            onset=float(time_onset),
                            offset=offset_value,
                            onpedal=0.0,
                            offpedal=0.0,
                            pitch=pitch_value,
                            velocity=velocity_value,
                            reonset=False,
                        )
                    )

            if (
                (len(notes) > 1)
                and (notes[len(notes) - 1].pitch == notes[len(notes) - 2].pitch)
                and (notes[len(notes) - 1].onset < notes[len(notes) - 2].offset)
            ):
                notes[len(notes) - 2].offset = notes[len(notes) - 1].onset

    a_onpedal_detect = detect_event(
        hop_sec, np.expand_dims(onpedal, axis=1), 0, thred_onpedal
    )
    a_offpedal_detect = detect_event(
        hop_sec, np.expand_dims(offpedal, axis=1), 0, thred_offpedal
    )

    pedals: List[Pedal] = []

    for time_onset, offset_value, pitch_value, velocity_value in process_label(
        midi_config,
        hop_sec,
        0,
        a_onpedal_detect,
        a_offpedal_detect,
        np.expand_dims(mpe_pedal, axis=1),
        thred_mpe_pedal,
        None,
        mode_offset,
    ):
        pedal = Pedal(
            onset=float(time_onset),
            offset=offset_value,
        )
        pedals.append(pedal)

        if (
            (len(pedals) > 1)
            and (pedals[len(pedals) - 1].onset < pedals[len(pedals) - 2].offset)
        ):
            pedals[len(pedals) - 2].offset = pedals[len(pedals) - 1].onset - 0.01

    print("pedals", pedals)

    notes = sorted(sorted(notes, key=lambda x: x.pitch), key=lambda x: x.onset)

    return notes, pedals

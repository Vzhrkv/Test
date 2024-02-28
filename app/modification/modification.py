from typing import Tuple
from librosa.effects import pitch_shift, time_stretch
from numpy import ndarray

from app.base import ReadAudio


ProcessedAudio = Tuple[ndarray, float]


def shift_pitch(
    readed_audio: ReadAudio,
    steps: int,
    bins_per_octave: int,
) -> ProcessedAudio:
    if bins_per_octave <= 0:
        raise ValueError(
            f"Bins per octave must be greater than zero(Given bins: {bins_per_octave})"
        )
    y, sr = readed_audio
    if steps == 0:
        return y, sr

    shifted_y = pitch_shift(
        y=y,
        sr=sr,
        n_steps=steps,
        bins_per_octave=bins_per_octave,
    )
    return shifted_y, sr


def stretch_audio(
    readed_audio: ReadAudio,
    rate: float,
) -> ProcessedAudio:
    if rate <= 0:
        raise ValueError(f"Time stretch can't be less than zero(Given rate: {rate}).")

    return time_stretch(y=readed_audio[0], rate=rate), readed_audio[1]

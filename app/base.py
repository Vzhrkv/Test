from os import PathLike
from pathlib import Path
from typing import Tuple, TypeAlias

from numpy import ndarray
import librosa


ReadedAudio: TypeAlias = Tuple[ndarray, float]


def read_audio(file: PathLike) -> ReadedAudio:
    pfile = Path(file)
    if not pfile.exists():
        raise FileNotFoundError(f"{pfile} not found.")

    audio_ts, sampling_rate = librosa.load(file)

    return audio_ts, sampling_rate

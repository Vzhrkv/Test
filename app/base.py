import json
import logging
from os import PathLike
from pathlib import Path
from typing import Tuple, TypeAlias

from numpy import ndarray
import librosa


ReadAudio: TypeAlias = Tuple[ndarray, float]


def read_audio(file: PathLike) -> ReadAudio:
    pfile = Path(file)
    if not pfile.exists():
        raise FileNotFoundError(f"{pfile} not found.")

    audio_ts, sampling_rate = librosa.load(file)

    return audio_ts, sampling_rate


class JSONFormatter(logging.Formatter):
    def __init__(self):
        super().__init__()

    def format(self, record):
        record.msg = json.dumps(
            record.msg,
            ensure_ascii=False,
        )
        return super().format(record)

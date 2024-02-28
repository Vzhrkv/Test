from os import PathLike
from pathlib import Path

from whisper import load_model, Whisper  # type: ignore

from app.base import read_audio


def transcribe(model_name: str, audio_file: PathLike) -> dict[str, str | list]:
    model = load_model(name=model_name)

    if not Path(audio_file).exists():
        raise FileNotFoundError()

    read_audio_ = read_audio(audio_file)

    result = model.transcribe(audio=read_audio_[0])
    return result

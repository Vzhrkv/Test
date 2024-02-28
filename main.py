#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import logging
import json
from pathlib import Path
from typing import Annotated, Optional

from soundfile import write  # type: ignore
import typer

from app.base import JSONFormatter, read_audio
from app.modification import shift_pitch, stretch_audio
from app.transcribition import transcribe


app = typer.Typer()

available_models = ["tiny", "base", "small", "medium", "large"]


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
loggingStreamHandler = logging.FileHandler("log.json")
loggingStreamHandler.setFormatter(JSONFormatter())
logger.addHandler(loggingStreamHandler)


@app.command(name="change")
def modificate(
    file_name: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="File to modificate. Supported only WAV format.",
        ),
    ],
    pitch_shift_steps: Annotated[
        Optional[int],
        typer.Option(
            "--shift",
            "-s",
            help="""Steps to move pitch. 
            If bins = 12 and pitch shift = 1, it's equal to one semitone""",
        ),
    ] = 0,
    time_stretch_rate: Annotated[
        Optional[float],
        typer.Option(
            "--rate",
            "-r",
            help="""Time stretch rate. Must be greater than zero. 
            If rate > 1, then audio sped up. Else slowed down by rate.""",
        ),
    ] = 1.0,
    bins_per_octave: Annotated[
        Optional[int],
        typer.Option(
            "--bins",
            "-b",
            help="""Number of bins per octave.""",
        ),
    ] = 12,
    output_file_name: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output file name.",
        ),
    ] = None,
):
    readed_audio = read_audio(file=file_name)
    shifted_pitch = shift_pitch(
        readed_audio=readed_audio,
        steps=pitch_shift_steps if pitch_shift_steps is not None else 0,
        bins_per_octave=bins_per_octave,  # type: ignore
    )
    stretched_audio, sample_rate = stretch_audio(
        readed_audio=shifted_pitch,
        rate=time_stretch_rate,  # type: ignore
    )

    if output_file_name is None:
        output_file_name = Path(f"samples/{file_name.stem}_processed{file_name.suffix}")

    write(
        file=(output_file_name if output_file_name is not None else f"{file_name}"),
        data=stretched_audio,
        samplerate=sample_rate,
    )


@app.command("totext")
def audio_to_text(
    file_name: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="File to modificate. Supported only WAV format.",
        ),
    ],
    model_name: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="""Model name. 
            Available models: tiny, base, small, medium, large
            """,
        ),
    ] = "tiny",
    output_file_name: Annotated[
        Optional[Path],
        typer.Option(
            "--file",
            "-f",
            help="Output file name.",
        ),
    ] = None,
):

    if model_name not in available_models:
        raise ValueError(
            f"Available models: {available_models}. Your model is: {model_name}."
        )
    result = transcribe(model_name=model_name, audio_file=file_name)

    logger.info(result)


if __name__ == "__main__":
    app()

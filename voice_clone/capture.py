from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
from requests import RequestException

from voice_clone.core import ReferenceAnalysis, prepare_reference
from voice_clone.transcription import transcribe


@dataclass(frozen=True)
class CaptureInspection:
    analysis: ReferenceAnalysis
    transcript: str
    transcription_error: str | None


def inspect_reference(source: str | Path, language: str) -> CaptureInspection:
    prepared = prepare_reference(source)
    try:
        audio, sample_rate = sf.read(prepared.path, dtype="float32")
        try:
            transcript = transcribe(audio, sample_rate, language)
        except (
            ImportError,
            OSError,
            RequestException,
            RuntimeError,
            ValueError,
        ) as exc:
            return CaptureInspection(
                analysis=prepared.analysis,
                transcript="",
                transcription_error=f"{type(exc).__name__}: {exc}",
            )
        return CaptureInspection(
            analysis=prepared.analysis,
            transcript=transcript,
            transcription_error=None,
        )
    finally:
        prepared.path.unlink(missing_ok=True)

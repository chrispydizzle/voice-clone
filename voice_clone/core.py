from __future__ import annotations

from dataclasses import dataclass
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf

MIN_REFERENCE_SECONDS = 3.0
MAX_REFERENCE_SECONDS = 60.0
MAX_TEXT_LENGTH = 2_000
VOICE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
QualityRating = Literal["good", "caution", "poor"]
CLIPPING_SAMPLE_THRESHOLD = 0.999


@dataclass(frozen=True)
class ReferenceAnalysis:
    duration_seconds: float
    rms_amplitude: float
    rms_dbfs: float
    peak_amplitude: float
    peak_dbfs: float
    clipping_ratio: float
    duration_rating: QualityRating
    level_rating: QualityRating
    clipping_rating: QualityRating


@dataclass(frozen=True)
class PreparedReference:
    """Normalized reference audio and its analysis.

    The returned ``path`` is a caller-owned temporary WAV. Callers must unlink
    it after use, even when later processing fails.
    """

    path: Path
    analysis: ReferenceAnalysis


def _dbfs(amplitude: float) -> float:
    return -math.inf if amplitude <= 0 else 20 * math.log10(amplitude)


def _duration_rating(seconds: float) -> QualityRating:
    return "good" if 8 <= seconds <= 30 else "caution"


def _level_rating(rms_dbfs: float) -> QualityRating:
    if -30 <= rms_dbfs <= -10:
        return "good"
    if -40 <= rms_dbfs < -30 or -10 < rms_dbfs <= -3:
        return "caution"
    return "poor"


def _clipping_rating(clipping_ratio: float) -> QualityRating:
    if clipping_ratio < 0.001:
        return "good"
    if clipping_ratio < 0.01:
        return "caution"
    return "poor"


def analyze_waveform(audio: np.ndarray, sample_rate: int) -> ReferenceAnalysis:
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim != 1:
        raise ValueError("Reference analysis requires mono audio.")
    if sample_rate <= 0 or len(samples) == 0:
        raise ValueError("Reference audio is empty or has an invalid sample rate.")

    duration = len(samples) / sample_rate
    rms = math.sqrt(float(np.mean(np.square(samples))))
    peak = float(np.max(np.abs(samples)))
    clipping_ratio = float(np.mean(np.abs(samples) >= CLIPPING_SAMPLE_THRESHOLD))
    rms_dbfs = _dbfs(rms)
    peak_dbfs = _dbfs(peak)

    return ReferenceAnalysis(
        duration_seconds=duration,
        rms_amplitude=rms,
        rms_dbfs=rms_dbfs,
        peak_amplitude=peak,
        peak_dbfs=peak_dbfs,
        clipping_ratio=clipping_ratio,
        duration_rating=_duration_rating(duration),
        level_rating=_level_rating(rms_dbfs),
        clipping_rating=_clipping_rating(clipping_ratio),
    )


def validate_voice_id(voice_id: str) -> str:
    if not isinstance(voice_id, str):
        raise ValueError("Select a saved voice or create a new one.")
    normalized = voice_id.strip()
    if not VOICE_ID_PATTERN.fullmatch(normalized):
        raise ValueError(
            "Voice name must be 1-40 characters using only letters, numbers, "
            "hyphens, or underscores."
        )
    return normalized


def validate_text(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("Enter some text to speak.")
    normalized = text.strip()
    if not normalized:
        raise ValueError("Enter some text to speak.")
    if len(normalized) > MAX_TEXT_LENGTH:
        raise ValueError(f"Text must be {MAX_TEXT_LENGTH:,} characters or fewer.")
    return normalized


def prepare_reference(source: str | Path) -> PreparedReference:
    """Normalize a reference clip and return the temp file plus analysis.

    The returned temporary WAV is owned by the caller and must be unlinked
    after the reference is consumed.
    """

    source_path = Path(source)
    if not source_path.is_file():
        raise ValueError("Record or upload a reference audio clip first.")

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required but was not found on PATH.")

    handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    normalized_path = Path(handle.name)
    handle.close()

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_path),
        "-ac",
        "1",
        "-ar",
        "24000",
        "-c:a",
        "pcm_s16le",
        str(normalized_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        normalized_path.unlink(missing_ok=True)
        detail = result.stderr.strip() or "unknown FFmpeg error"
        raise ValueError(f"Could not read the reference audio: {detail}")

    try:
        audio, sample_rate = sf.read(normalized_path, dtype="float32")
        analysis = analyze_waveform(audio, sample_rate)
    except Exception:
        normalized_path.unlink(missing_ok=True)
        raise

    if analysis.duration_seconds < MIN_REFERENCE_SECONDS:
        normalized_path.unlink(missing_ok=True)
        raise ValueError(
            f"Reference audio is {analysis.duration_seconds:.1f}s; record at least "
            f"{MIN_REFERENCE_SECONDS:.0f}s."
        )
    if analysis.duration_seconds > MAX_REFERENCE_SECONDS:
        normalized_path.unlink(missing_ok=True)
        raise ValueError(
            f"Reference audio is {analysis.duration_seconds:.1f}s; keep it under "
            f"{MAX_REFERENCE_SECONDS:.0f}s."
        )
    if analysis.rms_amplitude < 0.003:
        normalized_path.unlink(missing_ok=True)
        raise ValueError("The reference audio is silent or too quiet.")

    return PreparedReference(path=normalized_path, analysis=analysis)


def normalize_reference(source: str | Path) -> tuple[Path, float]:
    """Backward-compatible wrapper returning a caller-owned temp path."""

    prepared = prepare_reference(source)
    return prepared.path, prepared.analysis.duration_seconds

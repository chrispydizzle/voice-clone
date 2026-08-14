from __future__ import annotations

import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import soundfile as sf

MIN_REFERENCE_SECONDS = 3.0
MAX_REFERENCE_SECONDS = 60.0
MAX_TEXT_LENGTH = 2_000
VOICE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


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


def normalize_reference(source: str | Path) -> tuple[Path, float]:
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
        duration = len(audio) / sample_rate
        rms = math.sqrt(float((audio * audio).mean())) if len(audio) else 0.0
    except Exception:
        normalized_path.unlink(missing_ok=True)
        raise

    if duration < MIN_REFERENCE_SECONDS:
        normalized_path.unlink(missing_ok=True)
        raise ValueError(
            f"Reference audio is {duration:.1f}s; record at least "
            f"{MIN_REFERENCE_SECONDS:.0f}s."
        )
    if duration > MAX_REFERENCE_SECONDS:
        normalized_path.unlink(missing_ok=True)
        raise ValueError(
            f"Reference audio is {duration:.1f}s; keep it under "
            f"{MAX_REFERENCE_SECONDS:.0f}s."
        )
    if rms < 0.003:
        normalized_path.unlink(missing_ok=True)
        raise ValueError("The reference audio is silent or too quiet.")

    return normalized_path, duration

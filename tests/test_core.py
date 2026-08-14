from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voice_clone.core import normalize_reference, validate_text, validate_voice_id


@pytest.mark.parametrize("voice_id", ["my_voice", "Voice-2", "a"])
def test_validate_voice_id_accepts_safe_names(voice_id):
    assert validate_voice_id(voice_id) == voice_id


@pytest.mark.parametrize("voice_id", ["", "two words", "../voice", "x" * 41])
def test_validate_voice_id_rejects_unsafe_names(voice_id):
    with pytest.raises(ValueError):
        validate_voice_id(voice_id)


def test_validate_text_trims_input():
    assert validate_text("  Hello  ") == "Hello"


def test_validate_text_rejects_empty_input():
    with pytest.raises(ValueError):
        validate_text("   ")


def test_normalize_reference_accepts_audible_audio(tmp_path: Path):
    sample_rate = 24_000
    seconds = 4
    time = np.arange(sample_rate * seconds) / sample_rate
    audio = 0.1 * np.sin(2 * np.pi * 220 * time)
    source = tmp_path / "reference.wav"
    sf.write(source, audio, sample_rate)

    normalized, duration = normalize_reference(source)
    try:
        assert duration == pytest.approx(seconds, abs=0.05)
        info = sf.info(normalized)
        assert info.channels == 1
        assert info.samplerate == 24_000
    finally:
        normalized.unlink(missing_ok=True)


def test_normalize_reference_rejects_silence(tmp_path: Path):
    source = tmp_path / "silence.wav"
    sf.write(source, np.zeros(24_000 * 4), 24_000)

    with pytest.raises(ValueError, match="silent"):
        normalize_reference(source)


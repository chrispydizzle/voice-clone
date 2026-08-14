from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voice_clone.core import (
    analyze_waveform,
    normalize_reference,
    prepare_reference,
    validate_text,
    validate_voice_id,
)


def sine_wave(amplitude: float, seconds: int = 10, sample_rate: int = 24_000):
    time = np.arange(sample_rate * seconds) / sample_rate
    return amplitude * np.sin(2 * np.pi * 220 * time)


def test_analyze_waveform_rates_clean_reference_good():
    analysis = analyze_waveform(sine_wave(0.1), 24_000)

    assert analysis.duration_seconds == pytest.approx(10.0)
    assert analysis.duration_rating == "good"
    assert analysis.level_rating == "good"
    assert analysis.clipping_rating == "good"
    assert analysis.rms_dbfs == pytest.approx(-23.01, abs=0.1)
    assert analysis.clipping_ratio == 0.0


@pytest.mark.parametrize(
    ("audio", "expected_level", "expected_clipping"),
    [
        (sine_wave(0.005), "poor", "good"),
        (sine_wave(0.8), "caution", "good"),
        (np.ones(24_000 * 10, dtype=np.float32), "poor", "poor"),
    ],
)
def test_analyze_waveform_rates_problem_signals(
    audio, expected_level, expected_clipping
):
    analysis = analyze_waveform(audio, 24_000)

    assert analysis.level_rating == expected_level
    assert analysis.clipping_rating == expected_clipping


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(4, "caution"), (8, "good"), (30, "good"), (45, "caution")],
)
def test_analyze_waveform_rates_valid_durations(seconds, expected):
    analysis = analyze_waveform(sine_wave(0.1, seconds=seconds), 24_000)

    assert analysis.duration_rating == expected


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

    prepared = prepare_reference(source)
    normalized = None
    try:
        assert prepared.analysis.duration_seconds == pytest.approx(seconds, abs=0.05)
        assert prepared.analysis.level_rating == "good"
        info = sf.info(prepared.path)
        assert info.channels == 1
        assert info.samplerate == 24_000
        normalized, duration = normalize_reference(source)
        assert duration == pytest.approx(prepared.analysis.duration_seconds)
    finally:
        prepared.path.unlink(missing_ok=True)
        if normalized is not None:
            normalized.unlink(missing_ok=True)


def test_normalize_reference_rejects_silence(tmp_path: Path):
    source = tmp_path / "silence.wav"
    sf.write(source, np.zeros(24_000 * 4), 24_000)

    with pytest.raises(ValueError, match="silent"):
        normalize_reference(source)

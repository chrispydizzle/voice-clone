from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voice_clone import capture
from voice_clone.core import PreparedReference, ReferenceAnalysis


def analysis_fixture() -> ReferenceAnalysis:
    return ReferenceAnalysis(
        duration_seconds=10.0,
        rms_amplitude=0.1,
        rms_dbfs=-20.0,
        peak_amplitude=0.2,
        peak_dbfs=-13.98,
        clipping_ratio=0.0,
        duration_rating="good",
        level_rating="good",
        clipping_rating="good",
    )


def write_reference(path: Path) -> None:
    sf.write(path, np.zeros(24_000, dtype=np.float32), 24_000)


def test_inspect_reference_returns_transcript_and_deletes_temp(tmp_path, monkeypatch):
    normalized = tmp_path / "normalized.wav"
    write_reference(normalized)
    prepared = PreparedReference(normalized, analysis_fixture())
    monkeypatch.setattr(capture, "prepare_reference", lambda _: prepared)
    monkeypatch.setattr(capture, "transcribe", lambda audio, rate, language: "Hello")

    result = capture.inspect_reference("source.wav", "English")

    assert result.transcript == "Hello"
    assert result.transcription_error is None
    assert result.analysis == analysis_fixture()
    assert not normalized.exists()


def test_inspect_reference_surfaces_transcription_error_and_deletes_temp(
    tmp_path, monkeypatch
):
    normalized = tmp_path / "normalized.wav"
    write_reference(normalized)
    prepared = PreparedReference(normalized, analysis_fixture())
    monkeypatch.setattr(capture, "prepare_reference", lambda _: prepared)

    def fail_transcription(audio, rate, language):
        raise RuntimeError("GPU allocation failed")

    monkeypatch.setattr(capture, "transcribe", fail_transcription)

    result = capture.inspect_reference("source.wav", "English")

    assert result.transcript == ""
    assert result.transcription_error == "RuntimeError: GPU allocation failed"
    assert not normalized.exists()


def test_inspect_reference_does_not_hide_audio_validation_errors(monkeypatch):
    def fail_preparation(_):
        raise ValueError("Reference audio is silent or too quiet.")

    monkeypatch.setattr(capture, "prepare_reference", fail_preparation)

    with pytest.raises(ValueError, match="Reference audio is silent or too quiet."):
        capture.inspect_reference("source.wav", "English")


def test_inspect_reference_deletes_temp_when_audio_read_fails(tmp_path, monkeypatch):
    normalized = tmp_path / "normalized.wav"
    write_reference(normalized)
    prepared = PreparedReference(normalized, analysis_fixture())
    monkeypatch.setattr(capture, "prepare_reference", lambda _: prepared)
    monkeypatch.setattr(
        capture.sf,
        "read",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("read failed")),
    )

    with pytest.raises(RuntimeError, match="read failed"):
        capture.inspect_reference("source.wav", "English")

    assert not normalized.exists()

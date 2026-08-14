from voice_clone.capture import CaptureInspection
from voice_clone.core import ReferenceAnalysis

import app


def analysis_fixture() -> ReferenceAnalysis:
    return ReferenceAnalysis(
        duration_seconds=14.2,
        rms_amplitude=0.1,
        rms_dbfs=-20.0,
        peak_amplitude=0.2,
        peak_dbfs=-13.98,
        clipping_ratio=0.0,
        duration_rating="good",
        level_rating="good",
        clipping_rating="good",
    )


class FakeProgress:
    def __init__(self):
        self.calls = []

    def __call__(self, value, desc):
        self.calls.append((value, desc))


def test_format_reference_analysis_shows_all_metrics():
    html = app.format_reference_analysis(analysis_fixture())

    assert "Duration" in html
    assert "14.2s" in html
    assert "Input level" in html
    assert "-20.0 dBFS" in html
    assert "Clipping" in html
    assert "0.00%" in html
    assert 'data-rating="good"' in html


def test_inspect_reference_ui_returns_transcript_and_quality(monkeypatch):
    result = CaptureInspection(
        analysis=analysis_fixture(),
        transcript="Correct transcript.",
        transcription_error=None,
    )
    monkeypatch.setattr(app, "inspect_reference", lambda source, language: result)
    progress = FakeProgress()

    transcript, quality, status = app.inspect_reference_ui(
        "reference.wav", "English", progress
    )

    assert transcript == "Correct transcript."
    assert "14.2s" in quality
    assert "Transcription complete" in status
    assert progress.calls[-1] == (1.0, "Reference ready")


def test_inspect_reference_ui_keeps_manual_fallback(monkeypatch):
    result = CaptureInspection(
        analysis=analysis_fixture(),
        transcript="",
        transcription_error="RuntimeError: model unavailable",
    )
    monkeypatch.setattr(app, "inspect_reference", lambda source, language: result)

    transcript, quality, status = app.inspect_reference_ui(
        "reference.wav", "English", FakeProgress()
    )

    assert transcript == ""
    assert "14.2s" in quality
    assert "Enter the transcript manually" in status
    assert "model unavailable" in status


def test_inspect_reference_ui_clears_fields_without_audio():
    assert app.inspect_reference_ui(None, "English", FakeProgress()) == ("", "", "")


def test_reference_quality_advice_is_actionable():
    analysis = ReferenceAnalysis(
        duration_seconds=45.0,
        rms_amplitude=0.005,
        rms_dbfs=-46.0,
        peak_amplitude=1.0,
        peak_dbfs=0.0,
        clipping_ratio=0.02,
        duration_rating="caution",
        level_rating="poor",
        clipping_rating="poor",
    )

    advice = app.reference_quality_advice(analysis)

    assert "8-30 seconds" in advice
    assert "-30 to -10 dBFS" in advice
    assert "Lower microphone gain" in advice


def test_audio_component_preserves_microphone_and_upload_sources():
    config = app.demo.get_config_file()
    reference = next(
        component
        for component in config["components"]
        if component["props"].get("label")
        == "Reference recording (10-30 seconds recommended)"
    )

    assert set(reference["props"]["sources"]) == {"microphone", "upload"}

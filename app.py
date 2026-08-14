from __future__ import annotations

import gradio as gr

from voice_clone.capture import inspect_reference
from voice_clone.core import ReferenceAnalysis
from voice_clone.model import create_voice, delete_voice, list_voices, synthesize

LANGUAGES = {
    "English": "English",
    "Spanish": "Spanish",
    "French": "French",
    "German": "German",
    "Italian": "Italian",
    "Portuguese": "Portuguese",
    "Russian": "Russian",
    "Chinese": "Chinese",
    "Japanese": "Japanese",
    "Korean": "Korean",
}


def create_voice_ui(*args):
    try:
        audio, voice_id, status = create_voice(*args)
        return audio, gr.update(choices=list_voices(), value=voice_id), status
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def synthesize_ui(*args):
    try:
        return synthesize(*args)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def refresh_voices_ui(current_voice):
    voices = list_voices()
    selected = current_voice if current_voice in voices else (voices[0] if voices else None)
    return gr.update(choices=voices, value=selected)


def delete_voice_ui(voice_id):
    try:
        status = delete_voice(voice_id)
        voices = list_voices()
        return gr.update(choices=voices, value=voices[0] if voices else None), status
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


RATING_COPY = {
    "good": "Good",
    "caution": "Check",
    "poor": "Poor",
}


def _quality_card(label: str, value: str, rating: str) -> str:
    return (
        f'<div class="quality-card" data-rating="{rating}">'
        f'<span class="quality-label">{label}</span>'
        f"<strong>{value}</strong>"
        f"<span>{RATING_COPY[rating]}</span>"
        "</div>"
    )


def format_reference_analysis(analysis: ReferenceAnalysis) -> str:
    clipping_percent = analysis.clipping_ratio * 100
    return (
        '<div class="quality-grid">'
        + _quality_card(
            "Duration",
            f"{analysis.duration_seconds:.1f}s",
            analysis.duration_rating,
        )
        + _quality_card(
            "Input level",
            f"{analysis.rms_dbfs:.1f} dBFS",
            analysis.level_rating,
        )
        + _quality_card(
            "Clipping",
            f"{clipping_percent:.2f}%",
            analysis.clipping_rating,
        )
        + "</div>"
    )


def reference_quality_advice(analysis: ReferenceAnalysis) -> str:
    advice = []
    if analysis.duration_rating != "good":
        advice.append("For best similarity, use 8-30 seconds of speech.")
    if analysis.level_rating != "good":
        advice.append(
            "Adjust microphone distance or gain toward -30 to -10 dBFS."
        )
    if analysis.clipping_rating != "good":
        advice.append("Lower microphone gain and record again to reduce clipping.")
    return " ".join(advice)


def _inspect_reference_outputs(reference_audio, language, progress):
    if not reference_audio:
        return None

    progress(0.1, desc="Preparing reference audio")
    try:
        result = inspect_reference(reference_audio, language)
    except (OSError, RuntimeError, ValueError) as exc:
        raise gr.Error(str(exc)) from exc

    quality = format_reference_analysis(result.analysis)
    if result.transcription_error:
        status_parts = [
            "Automatic transcription failed. Enter the transcript manually, "
            f"then create the voice. `{result.transcription_error}`"
        ]
    else:
        status_parts = [
            "Transcription complete. Correct any mistakes before cloning."
        ]
    advice = reference_quality_advice(result.analysis)
    if advice:
        status_parts.append(advice)
    progress(1.0, desc="Reference ready")
    return result, quality, " ".join(status_parts)


def inspect_reference_ui(
    reference_audio,
    language,
    progress=gr.Progress(),
):
    outputs = _inspect_reference_outputs(reference_audio, language, progress)
    if outputs is None:
        return "", "", ""

    result, quality, status = outputs
    return result.transcript, quality, status


def retry_reference_ui(
    reference_audio,
    language,
    current_transcript,
    progress=gr.Progress(),
):
    outputs = _inspect_reference_outputs(reference_audio, language, progress)
    if outputs is None:
        return "", "", ""

    result, quality, status = outputs
    transcript = current_transcript if result.transcription_error else result.transcript
    return transcript, quality, status


CSS = """
.gradio-container { max-width: 1050px !important; }
.hero { text-align: center; margin-bottom: 1rem; }
.hero h1 { font-size: 2.25rem; margin-bottom: .25rem; }
.step { border: 1px solid var(--border-color-primary); border-radius: 14px; padding: 18px; }
.quality-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
.quality-card { border: 1px solid var(--border-color-primary); border-radius: 10px; padding: 10px; display: grid; gap: 3px; }
.quality-card[data-rating="good"] { border-color: #22c55e; }
.quality-card[data-rating="caution"] { border-color: #f59e0b; }
.quality-card[data-rating="poor"] { border-color: #ef4444; }
.quality-label { font-size: 0.8rem; opacity: 0.75; }
"""


with gr.Blocks(title="Local Voice Clone", css=CSS) as demo:
    gr.HTML(
        """
        <div class="hero">
          <h1>Local Voice Clone</h1>
          <p>Record or upload once, verify the transcript, then speak any text.</p>
        </div>
        """
    )

    with gr.Row(equal_height=False):
        with gr.Column(elem_classes="step"):
            gr.Markdown("## 1. Capture")
            reference = gr.Audio(
                label="Reference recording (10-30 seconds recommended)",
                sources=["microphone", "upload"],
                type="filepath",
            )
            quality_summary = gr.HTML()
            transcription_status = gr.Markdown()

        with gr.Column(elem_classes="step"):
            gr.Markdown("## 2. Verify and create")
            language = gr.Dropdown(
                label="Language",
                choices=list(LANGUAGES),
                value="English",
                type="value",
            )
            reference_text = gr.Textbox(
                label="Reference transcript",
                placeholder="Automatic transcription appears here. Correct it exactly.",
                lines=4,
                max_length=2000,
            )
            transcribe_again = gr.Button("Transcribe again")
            voice_name = gr.Textbox(label="Voice name", value="my_voice", max_lines=1)
            owns_voice = gr.Checkbox(
                label="I own this voice or have explicit permission to clone it"
            )
            create_button = gr.Button("Create voice", variant="primary")
            preview = gr.Audio(label="Voice preview", type="filepath")
            create_status = gr.Markdown()

    with gr.Column(elem_classes="step"):
        gr.Markdown("## 3. Type and speak")
        with gr.Row():
            saved_voice = gr.Dropdown(
                label="Saved voice",
                choices=list_voices(),
                value=list_voices()[0] if list_voices() else None,
                scale=3,
            )
            refresh_button = gr.Button("Refresh", scale=1)
        text = gr.Textbox(
            label="Text",
            placeholder="Type what you want your cloned voice to say...",
            lines=6,
            max_length=2000,
        )
        with gr.Row():
            speak_button = gr.Button("Speak", variant="primary")
            delete_button = gr.Button("Delete selected voice", variant="stop")
        generated = gr.Audio(label="Generated speech", type="filepath")
        speak_status = gr.Markdown()

    gr.Markdown(
        "Powered locally by Qwen3-TTS 1.7B Base (Apache 2.0). Do not use cloned "
        "speech to impersonate, deceive, or misrepresent another person."
    )

    reference.change(
        fn=inspect_reference_ui,
        inputs=[reference, language],
        outputs=[reference_text, quality_summary, transcription_status],
    )
    transcribe_again.click(
        fn=retry_reference_ui,
        inputs=[reference, language, reference_text],
        outputs=[reference_text, quality_summary, transcription_status],
    )
    create_button.click(
        fn=create_voice_ui,
        inputs=[
            reference,
            reference_text,
            voice_name,
            language,
            owns_voice,
        ],
        outputs=[preview, saved_voice, create_status],
    )
    speak_button.click(
        fn=synthesize_ui,
        inputs=[text, saved_voice, language],
        outputs=[generated, speak_status],
    )
    refresh_button.click(
        fn=refresh_voices_ui,
        inputs=[saved_voice],
        outputs=[saved_voice],
    )
    delete_button.click(
        fn=delete_voice_ui,
        inputs=[saved_voice],
        outputs=[saved_voice, speak_status],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        inbrowser=True,
        server_name="127.0.0.1",
        share=False,
    )

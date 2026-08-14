from __future__ import annotations

import gradio as gr

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


CSS = """
.gradio-container { max-width: 1050px !important; }
.hero { text-align: center; margin-bottom: 1rem; }
.hero h1 { font-size: 2.25rem; margin-bottom: .25rem; }
.step { border: 1px solid var(--border-color-primary); border-radius: 14px; padding: 18px; }
"""


with gr.Blocks(title="Local Voice Clone", css=CSS) as demo:
    gr.HTML(
        """
        <div class="hero">
          <h1>Local Voice Clone</h1>
          <p>Record once, then turn typed text into speech in your voice.</p>
        </div>
        """
    )

    with gr.Row(equal_height=False):
        with gr.Column(elem_classes="step"):
            gr.Markdown("## 1. Create your voice")
            reference = gr.Audio(
                label="Reference recording (10-30 seconds recommended)",
                sources=["microphone", "upload"],
                type="filepath",
            )
            reference_text = gr.Textbox(
                label="Exact transcript of the reference recording",
                placeholder="Type exactly what you said, including contractions and punctuation.",
                lines=3,
                max_length=2000,
            )
            voice_name = gr.Textbox(
                label="Voice name",
                value="my_voice",
                max_lines=1,
            )
            language = gr.Dropdown(
                label="Language",
                choices=list(LANGUAGES),
                value="English",
                type="value",
            )
            owns_voice = gr.Checkbox(
                label="I own this voice or have explicit permission to clone it"
            )
            create_button = gr.Button("Create voice", variant="primary")
            preview = gr.Audio(label="Voice preview", type="filepath")
            create_status = gr.Markdown()

        with gr.Column(elem_classes="step"):
            gr.Markdown("## 2. Type and speak")
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
                lines=8,
                max_length=2000,
            )
            speak_button = gr.Button("Speak", variant="primary")
            delete_button = gr.Button("Delete selected voice", variant="stop")
            generated = gr.Audio(label="Generated speech", type="filepath")
            speak_status = gr.Markdown()

    gr.Markdown(
        "Powered locally by Qwen3-TTS 1.7B Base (Apache 2.0). Do not use cloned "
        "speech to impersonate, deceive, or misrepresent another person."
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

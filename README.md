# Local Voice Clone

Create reusable local voice profiles from a recording, then type text and
generate speech with that voice. The app runs on your computer and binds only
to `127.0.0.1`; after the Qwen3-TTS and Whisper Small models have been
downloaded, cloning and transcription work offline.

> **Use responsibly.** Clone only your own voice or a voice you have explicit
> permission to use. Do not use generated speech to impersonate, deceive, or
> misrepresent another person.

## Features

- Record a reference clip with a microphone or upload an existing audio file.
- Transcribe the reference locally with Whisper Small and correct the
  transcript before cloning.
- Get duration, level, and clipping feedback to improve voice similarity.
- Save named voice profiles locally and reuse them after restarting the app.
- Generate preview speech and new typed text with Qwen3-TTS.
- Keep the app local: sharing is disabled and the server binds to localhost.

## Quick start

### Requirements

- Windows 10 or 11
- 64-bit Python 3.12, available through the Python Launcher (`py`)
- An NVIDIA GPU is recommended. CPU-only setup is supported but much slower.
- Internet access for first-time setup and model downloads

The setup script installs FFmpeg and SoX through WinGet when they are not
already available.

### Install and run

```powershell
git clone https://github.com/chrispydizzle/voice-clone.git
cd voice-clone
.\setup.ps1
.\run.ps1
```

The browser opens to `http://127.0.0.1:7860`. For a CPU-only PyTorch install,
run:

```powershell
.\setup.ps1 -Cpu
```

## Create and use a voice

1. Record or upload a clear reference clip. Eight to 30 seconds is
   recommended; clips from three through 60 seconds are accepted.
2. Review the automatic transcript and correct it exactly before creating the
   voice.
3. Enter a voice name, confirm that you own the voice or have permission, and
   select **Create voice**.
4. Choose the saved voice in **Type and speak**, enter text, and select
   **Speak**.

Reference recordings are normalized temporarily for processing and then
deleted. Raw files held by Gradio's local cache are cleaned up after one hour.
Saved profiles in `voices\` contain voice embeddings and the corrected
reference transcript; generated WAV files are written to `outputs\`. These
directories are ignored by Git but are not encrypted.

## Configuration

| Variable | Purpose |
| --- | --- |
| `VOICE_CLONE_DEVICE` | Selects the device for Qwen cloning and synthesis, for example `cuda:0` or `cpu`. |
| `VOICE_CLONE_TRANSCRIBE_DEVICE` | Overrides the device used by Whisper transcription. |

If neither variable is set, the app uses `cuda:0` when CUDA is available and
otherwise uses the CPU. GPU failures are surfaced instead of silently falling
back to CPU.

## Development

Set up the development environment and run the repository checks:

```powershell
.\setup.ps1 -Dev
.\check.ps1
```

The test suite does not download model weights and does not require a GPU.
See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution and manual
voice-wizard verification guidance.

## Security and privacy

Report vulnerabilities through GitHub's private vulnerability reporting; see
[SECURITY.md](SECURITY.md). Read [SAFETY.md](SAFETY.md) before using the app,
especially if recordings or generated speech involve other people.

## License

This project is licensed under the [MIT License](LICENSE). Model and runtime
attributions are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

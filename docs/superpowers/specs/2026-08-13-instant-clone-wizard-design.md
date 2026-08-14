# Instant Clone Wizard Design

**Status:** Approved  
**Date:** 2026-08-13

## Goal

Make creating a high-quality reusable voice feel nearly automatic while keeping
the application fully local. A user should be able to record or upload one
clean sample, receive an editable transcript and useful quality feedback, and
create a persistent Qwen voice without manually transcribing the sample first.

## Product Scope

This milestone adds:

- Local automatic transcription for microphone recordings and uploaded audio.
- An editable transcript before voice creation.
- Deterministic quality indicators for duration, input level, and clipping.
- Clear progress, warning, and failure states.
- An enhanced two-panel creation layout.
- Preservation of the existing saved-voice and Type-and-Speak workflows.

This milestone does not add:

- Noise removal or automatic audio enhancement.
- Multi-take recording or multi-reference voice creation.
- Live voice conversion or streaming playback.
- Output galleries, projects, or generation history.
- Emotion, prosody, or advanced sampling controls.

## User Experience

### Upper workspace: voice creation

The upper workspace contains two adjacent panels.

#### Capture panel

The existing Gradio audio component remains the source control and continues to
support both microphone recording and file upload. The two sources follow the
same processing path.

When audio changes, the application automatically:

1. Normalizes it for analysis.
2. Measures duration, RMS level, peak level, and clipping ratio.
3. Runs local speech transcription.
4. Deletes the temporary normalized file.
5. Displays the transcript and quality results.

The panel shows three compact indicators:

- **Duration:** good from 8 to 30 seconds, caution from 3 to under 8 seconds or
  over 30 to 60 seconds, and blocking outside the existing 3-to-60-second
  limits.
- **Input level:** derived from RMS dBFS, with a warning when the signal is too
  quiet or close to full scale. Good is -30 through -10 dBFS; caution is -40
  through under -30 dBFS or above -10 through -3 dBFS; poor is below -40 dBFS
  or above -3 dBFS.
- **Clipping:** good below 0.1 percent clipped samples, caution from 0.1 to
  under 1 percent, and poor at 1 percent or more.

These indicators provide actionable advice but do not claim to predict clone
quality.

#### Voice setup panel

The adjacent panel contains:

- The automatically populated, editable reference transcript.
- Language selection.
- A Transcribe Again action for retrying after a language change or failure.
- Voice name.
- The existing ownership or permission confirmation.
- The Create Voice action.
- The generated preview and status.

The user may correct punctuation, contractions, names, and transcription errors
before creating the voice. If automatic transcription fails, manual transcript
entry remains available and voice creation continues to work. Changing the
language does not overwrite an edited transcript automatically; the user
explicitly selects Transcribe Again when a retry is wanted.

### Lower workspace: Type and Speak

The existing Type-and-Speak controls move below the creation workspace at full
width. Saved voice selection, refresh, text entry, synthesis, audio download,
and voice deletion retain their current behavior. A newly created voice is
automatically selected.

## Architecture

### `voice_clone/transcription.py`

Add a focused transcription service with:

- `TRANSCRIPTION_MODEL_NAME = "openai/whisper-small"`.
- A lazy singleton model and processor.
- A model-loading lock.
- A `transcribe(audio, language)` entry point returning normalized transcript
  text.
- CUDA inference using `torch.float16`; CPU fallback using `torch.float32`.
- Optional `VOICE_CLONE_TRANSCRIBE_DEVICE` override, defaulting to the main
  voice-clone device.

Use the Transformers automatic speech-recognition stack already required by
Qwen TTS. Pass an in-memory waveform loaded with SoundFile rather than a file
path so transcription does not depend on TorchCodec audio loading. Resample the
normalized waveform to Whisper's required sample rate before inference.

### `voice_clone/runtime.py`

Add one process-wide accelerator lock shared by transcription and Qwen
inference. This keeps GPU work serialized even if the Gradio queue or a future
API allows concurrent requests.

Model initialization remains independently locked so each model loads once.
The Whisper and Qwen models may remain resident after their first use; their
combined expected footprint fits the current 12 GB target GPU. The
transcription device override provides a CPU option for lower-memory systems.

### `voice_clone/core.py`

Introduce an immutable reference analysis result containing:

- Duration in seconds.
- RMS amplitude and RMS dBFS.
- Peak amplitude and peak dBFS.
- Clipping ratio.

Refactor normalization and analysis so transcription and voice creation reuse
the same validation rules. Every temporary file must be removed on success and
failure.

### `app.py`

Reorganize the Gradio layout and add an audio-change callback that:

1. Reports transcription progress.
2. Calls reference analysis and transcription.
3. Populates the transcript.
4. Renders the three quality indicators and any warning.

The existing creation callback continues to accept the source audio and the
user-corrected transcript. Existing synthesis and persistent voice callbacks
remain behaviorally unchanged.

## Data Flow

1. The user records audio or uploads a supported file.
2. Gradio supplies the source path to the analysis/transcription callback.
3. The callback creates a temporary normalized WAV.
4. Core analysis calculates objective audio metrics.
5. Whisper transcribes the waveform locally using the selected language.
6. The temporary WAV is deleted in a `finally` path.
7. The UI displays the editable transcript and quality indicators.
8. The user corrects the transcript and creates the voice.
9. Qwen extracts the reusable clone prompt and persists it in `voices\`.
10. The newly saved voice is selected for preview and later synthesis.

The raw reference recording is never copied into persistent application
storage.

## Failure Handling

- **No audio:** clear the transcript and quality state without starting a model.
- **Unreadable audio:** show the existing conversion error and do not
  transcribe.
- **Invalid duration or silence:** retain the existing blocking validation.
- **Poor level or clipping:** show an actionable warning but allow voice
  creation.
- **Whisper download or load failure:** show the underlying error and leave
  manual transcript entry enabled.
- **Transcription inference failure:** preserve the selected audio, clear only
  the generated transcript, and explain that manual entry is available.
- **GPU memory or runtime failure:** surface the error; do not silently move a
  model to CPU.

## Privacy and Storage

- Transcription and synthesis run locally after model weights are downloaded.
- No audio, transcript, or generated speech is sent to an external service.
- Temporary normalized audio is deleted after each operation.
- Only the existing Qwen prompt payload is persisted for reusable voices.
- Whisper weights use the Hugging Face user cache.

## Testing

Automated tests must not download model weights.

### Unit tests

- Clean, quiet, hot, and clipped waveform metric classification.
- Existing minimum and maximum duration validation.
- Language mapping passed to Whisper.
- Transcript whitespace normalization.
- Lazy model initialization and configured device selection.
- Successful and failed transcription with a mocked model.
- Temporary-file deletion after success and every failure path.
- UI result formatting for good, caution, and poor recordings.

### Regression tests

- Voice creation still accepts a manually entered transcript.
- Uploaded files and microphone-produced paths enter the same callback.
- Existing saved voices remain listable, loadable, synthesizable, and
  deletable.
- Existing synthesis remains serialized with transcription.

### Manual acceptance

1. Record a 10-to-20-second microphone sample and confirm transcription,
   editable text, indicators, voice creation, and preview.
2. Repeat with an uploaded WAV or common compressed audio file.
3. Deliberately use quiet and clipped samples and confirm useful warnings.
4. Simulate transcription failure and confirm manual transcript entry works.
5. Restart the application and generate speech with the saved voice.
6. Disconnect networking after model caching and confirm the complete workflow
   still functions.

## Acceptance Criteria

- Both microphone capture and file upload automatically produce an editable
  local transcript.
- The UI displays objective duration, level, and clipping feedback.
- Transcription failure never prevents manual voice creation.
- Raw reference audio is not added to persistent application storage.
- Existing saved voice and synthesis behavior remains intact.
- Automated tests pass without downloading Whisper or Qwen weights.
- After weights are cached, the feature works without network access.

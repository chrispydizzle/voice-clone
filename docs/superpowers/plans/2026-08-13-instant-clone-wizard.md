# Instant Clone Wizard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add fully local automatic reference transcription and objective audio-quality feedback while preserving microphone input, file upload, persistent voices, and existing synthesis behavior.

**Architecture:** Extend the audio core with pure analysis primitives, centralize GPU serialization in a runtime module, and add a lazy Whisper transcription service plus a capture orchestration layer. Keep Gradio presentation logic in `app.py`, with model-weight downloads excluded from automated tests through dependency injection and mocks.

**Tech Stack:** Python 3.12, Gradio 5.44.1, PyTorch/Torchaudio 2.8.0, Transformers 4.57.3, OpenAI Whisper Small, Qwen3-TTS 1.7B Base, SoundFile, NumPy, pytest.

## Global Constraints

- Support both Gradio microphone recording and file upload through the same processing path.
- Use `openai/whisper-small` for local multilingual transcription.
- Default transcription to the main voice-clone device; honor `VOICE_CLONE_TRANSCRIBE_DEVICE` when set.
- Use `torch.float16` for Whisper on CUDA and `torch.float32` on CPU.
- Keep the existing hard reference limits of 3 through 60 seconds.
- Rate duration as good from 8 through 30 seconds and caution inside the valid range outside that interval.
- Rate RMS level as good from -30 through -10 dBFS, caution from -40 through under -30 dBFS or above -10 through -3 dBFS, and poor below -40 dBFS or above -3 dBFS.
- Rate clipping as good below 0.1 percent, caution from 0.1 through under 1 percent, and poor at 1 percent or more.
- Do not persist raw reference recordings or temporary normalized audio.
- Do not silently move failed GPU work to CPU.
- Automated tests must not download Whisper or Qwen model weights.
- Preserve current saved-voice file compatibility and Type-and-Speak behavior.

---

## Planned File Structure

- Create `voice_clone/runtime.py`: shared device resolution and process-wide accelerator lock.
- Create `voice_clone/transcription.py`: lazy Whisper model ownership, resampling, language mapping, and transcript generation.
- Create `voice_clone/capture.py`: one orchestration boundary for prepare, analyze, transcribe, error reporting, and cleanup.
- Modify `voice_clone/core.py`: immutable analysis types, pure metric classification, and prepared-reference API.
- Modify `voice_clone/model.py`: use the shared accelerator lock without changing saved voice serialization.
- Modify `app.py`: enhanced two-panel capture/setup layout, automatic callback, retry action, and quality presentation.
- Modify `tests/test_core.py`: audio metric and prepared-reference tests.
- Create `tests/test_runtime.py`: device override and fallback tests.
- Create `tests/test_transcription.py`: mocked Whisper service tests.
- Create `tests/test_capture.py`: success, failure, and cleanup tests.
- Create `tests/test_app.py`: callback, presentation, and microphone/upload regression tests.
- Modify `THIRD_PARTY_NOTICES.md`, `SAFETY.md`, and `CONTRIBUTING.md`: model, privacy, and acceptance documentation.

---

### Task 1: Reference Audio Analysis

**Files:**
- Modify: `voice_clone/core.py:1-100`
- Modify: `tests/test_core.py:1-55`

**Interfaces:**
- Consumes: Existing FFmpeg normalization and `MIN_REFERENCE_SECONDS` / `MAX_REFERENCE_SECONDS`.
- Produces:
  - `QualityRating = Literal["good", "caution", "poor"]`
  - `ReferenceAnalysis`
  - `PreparedReference`
  - `analyze_waveform(audio: np.ndarray, sample_rate: int) -> ReferenceAnalysis`
  - `prepare_reference(source: str | Path) -> PreparedReference`
  - Backward-compatible `normalize_reference(source) -> tuple[Path, float]`

- [ ] **Step 1: Write failing metric-classification tests**

Add these imports and tests to `tests/test_core.py`:

```python
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
```

- [ ] **Step 2: Run the focused tests and confirm the missing API failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_core.py -q
```

Expected: collection fails because `analyze_waveform` and `prepare_reference` do not exist.

- [ ] **Step 3: Implement immutable metrics and classification**

Add the imports, types, constants, and pure analysis function to
`voice_clone/core.py`:

```python
from dataclasses import dataclass
from typing import Literal

import numpy as np

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
```

- [ ] **Step 4: Refactor normalization into `prepare_reference` without breaking callers**

Replace the post-FFmpeg section of `normalize_reference` with these functions:

```python
def prepare_reference(source: str | Path) -> PreparedReference:
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
    prepared = prepare_reference(source)
    return prepared.path, prepared.analysis.duration_seconds
```

Update the existing normalization test to assert `prepare_reference` metadata
and always unlink `prepared.path`.

- [ ] **Step 5: Run core tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_core.py -q
```

Expected: all `test_core.py` tests pass.

- [ ] **Step 6: Commit the analysis boundary**

```powershell
git add voice_clone\core.py tests\test_core.py
git commit -m "feat: analyze reference audio quality" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>" -m "Copilot-Session: ce8fca23-6d56-42c5-be2d-4c420668e8cf"
```

---

### Task 2: Shared Device Resolution and GPU Serialization

**Files:**
- Create: `voice_clone/runtime.py`
- Create: `tests/test_runtime.py`
- Modify: `voice_clone/model.py:1-259`
- Modify: `tests/test_model.py:1-89`

**Interfaces:**
- Consumes: PyTorch CUDA availability and the existing `VOICE_CLONE_DEVICE`.
- Produces:
  - `ACCELERATOR_LOCK`
  - `resolve_device(override_env: str | None = None) -> str`
  - Existing `model._device()` retained as a compatibility wrapper.

- [ ] **Step 1: Write failing runtime tests**

Create `tests/test_runtime.py`:

```python
import torch

from voice_clone import runtime


def test_resolve_device_uses_cuda_zero_by_default(monkeypatch):
    monkeypatch.delenv("VOICE_CLONE_DEVICE", raising=False)
    monkeypatch.delenv("VOICE_CLONE_TRANSCRIBE_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert runtime.resolve_device() == "cuda:0"


def test_resolve_device_uses_main_override(monkeypatch):
    monkeypatch.setenv("VOICE_CLONE_DEVICE", "cpu")
    monkeypatch.delenv("VOICE_CLONE_TRANSCRIBE_DEVICE", raising=False)

    assert runtime.resolve_device() == "cpu"


def test_resolve_device_prefers_specific_override(monkeypatch):
    monkeypatch.setenv("VOICE_CLONE_DEVICE", "cuda:0")
    monkeypatch.setenv("VOICE_CLONE_TRANSCRIBE_DEVICE", "cpu")

    assert runtime.resolve_device("VOICE_CLONE_TRANSCRIBE_DEVICE") == "cpu"
```

- [ ] **Step 2: Run the runtime tests and confirm the missing module failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_runtime.py -q
```

Expected: collection fails because `voice_clone.runtime` does not exist.

- [ ] **Step 3: Add the runtime ownership module**

Create `voice_clone/runtime.py`:

```python
from __future__ import annotations

import os
import threading

ACCELERATOR_LOCK = threading.Lock()


def resolve_device(override_env: str | None = None) -> str:
    requested = os.getenv(override_env) if override_env else None
    requested = requested or os.getenv("VOICE_CLONE_DEVICE")
    if requested:
        return requested

    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"
```

- [ ] **Step 4: Refactor Qwen inference to use the shared lock**

In `voice_clone/model.py`:

```python
from voice_clone.runtime import ACCELERATOR_LOCK, resolve_device
```

Remove `_inference_lock = threading.Lock()`, keep `_model_lock`, and replace
`_device` with:

```python
def _device() -> str:
    return resolve_device()
```

Replace both `with _inference_lock:` blocks with:

```python
with ACCELERATOR_LOCK:
```

Do not change voice payload serialization, model loading, or public function
signatures.

- [ ] **Step 5: Add a regression test proving synthesis takes the shared lock**

Add `import numpy as np` to `tests/test_model.py`, then append:

```python
class RecordingLock:
    def __init__(self):
        self.entered = False

    def __enter__(self):
        self.entered = True

    def __exit__(self, exc_type, exc, traceback):
        return False


def test_synthesize_uses_shared_accelerator_lock(tmp_path, monkeypatch):
    lock = RecordingLock()
    profile = model.VoiceProfile(prompt=["prompt"], language="English")

    class FakeQwenModel:
        def generate_voice_clone(self, **kwargs):
            assert lock.entered is True
            return [np.zeros(24, dtype=np.float32)], 24_000

    monkeypatch.setattr(model, "_load_voice_profile", lambda _: profile)
    monkeypatch.setattr(model, "_get_model", lambda: FakeQwenModel())
    monkeypatch.setattr(model, "_output_path", lambda _: tmp_path / "output.wav")
    monkeypatch.setattr(model, "ACCELERATOR_LOCK", lock)
    monkeypatch.setattr(model.sf, "write", lambda *args: None)

    output, _ = model.synthesize("Hello", "saved_voice", "English")

    assert output.endswith("output.wav")
    assert lock.entered is True
```

- [ ] **Step 6: Run runtime and model tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_runtime.py tests\test_model.py -q
```

Expected: all runtime and model tests pass.

- [ ] **Step 7: Commit shared runtime ownership**

```powershell
git add voice_clone\runtime.py voice_clone\model.py tests\test_runtime.py tests\test_model.py
git commit -m "refactor: share accelerator runtime ownership" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>" -m "Copilot-Session: ce8fca23-6d56-42c5-be2d-4c420668e8cf"
```

---

### Task 3: Lazy Local Whisper Transcription

**Files:**
- Create: `voice_clone/transcription.py`
- Create: `tests/test_transcription.py`

**Interfaces:**
- Consumes:
  - `ACCELERATOR_LOCK`
  - `resolve_device("VOICE_CLONE_TRANSCRIBE_DEVICE")`
  - Mono NumPy waveform and source sample rate.
- Produces:
  - `TRANSCRIPTION_MODEL_NAME`
  - `SUPPORTED_LANGUAGES`
  - `TranscriberBundle`
  - `normalize_transcript(text: str) -> str`
  - `transcribe(audio: np.ndarray, sample_rate: int, language: str) -> str`

- [ ] **Step 1: Write failing transcription tests with fake model objects**

Create `tests/test_transcription.py`:

```python
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from voice_clone import model
from voice_clone import transcription


class FakeProcessor:
    def __init__(self):
        self.seen_audio = None
        self.seen_sample_rate = None
        self.decoded_ids = None

    def __call__(self, audio, sampling_rate, return_tensors):
        self.seen_audio = audio
        self.seen_sample_rate = sampling_rate
        assert return_tensors == "pt"
        return SimpleNamespace(input_features=torch.ones(1, 80, 10))

    def batch_decode(self, generated_ids, skip_special_tokens):
        self.decoded_ids = generated_ids
        assert skip_special_tokens is True
        return ["  Hello,\n   local world.  "]


class FakeModel:
    def __init__(self):
        self.generate_kwargs = None

    def generate(self, input_features, **kwargs):
        assert input_features.shape == (1, 80, 10)
        self.generate_kwargs = kwargs
        return torch.tensor([[1, 2, 3]])


class RecordingLock:
    def __init__(self):
        self.entered = False

    def __enter__(self):
        self.entered = True

    def __exit__(self, exc_type, exc, traceback):
        return False


def test_normalize_transcript_collapses_whitespace():
    assert transcription.normalize_transcript("  one\n  two  ") == "one two"


def test_transcribe_uses_language_and_shared_lock(monkeypatch):
    processor = FakeProcessor()
    model = FakeModel()
    lock = RecordingLock()
    bundle = transcription.TranscriberBundle(
        model=model,
        processor=processor,
        device="cpu",
        dtype=torch.float32,
    )
    monkeypatch.setattr(transcription, "_get_transcriber", lambda: bundle)
    monkeypatch.setattr(transcription, "ACCELERATOR_LOCK", lock)

    text = transcription.transcribe(
        np.zeros(16_000, dtype=np.float32), 16_000, "English"
    )

    assert text == "Hello, local world."
    assert processor.seen_sample_rate == 16_000
    assert model.generate_kwargs == {"language": "english", "task": "transcribe"}
    assert lock.entered is True


def test_transcribe_rejects_unsupported_language():
    with pytest.raises(ValueError, match="Unsupported transcription language"):
        transcription.transcribe(
            np.zeros(16_000, dtype=np.float32), 16_000, "Klingon"
        )


def test_get_transcriber_builds_once(monkeypatch):
    sentinel = object()
    calls = []
    monkeypatch.setattr(transcription, "_transcriber", None)
    monkeypatch.setattr(
        transcription,
        "_build_transcriber",
        lambda: calls.append("build") or sentinel,
    )

    assert transcription._get_transcriber() is sentinel
    assert transcription._get_transcriber() is sentinel
    assert calls == ["build"]


def test_transcription_and_qwen_share_accelerator_lock():
    assert transcription.ACCELERATOR_LOCK is model.ACCELERATOR_LOCK
```

- [ ] **Step 2: Run the tests and confirm the missing module failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_transcription.py -q
```

Expected: collection fails because `voice_clone.transcription` does not exist.

- [ ] **Step 3: Implement language mapping, lazy loading, and dtype selection**

Create `voice_clone/transcription.py` with:

```python
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torchaudio.functional as audio_functional

from voice_clone.runtime import ACCELERATOR_LOCK, resolve_device

TRANSCRIPTION_MODEL_NAME = "openai/whisper-small"
WHISPER_SAMPLE_RATE = 16_000
SUPPORTED_LANGUAGES = {
    "English": "english",
    "Spanish": "spanish",
    "French": "french",
    "German": "german",
    "Italian": "italian",
    "Portuguese": "portuguese",
    "Russian": "russian",
    "Chinese": "chinese",
    "Japanese": "japanese",
    "Korean": "korean",
}


@dataclass(frozen=True)
class TranscriberBundle:
    model: Any
    processor: Any
    device: str
    dtype: torch.dtype


_transcriber: TranscriberBundle | None = None
_transcriber_lock = threading.Lock()


def _transcription_device() -> str:
    return resolve_device("VOICE_CLONE_TRANSCRIBE_DEVICE")


def _transcription_dtype(device: str) -> torch.dtype:
    return torch.float16 if device.startswith("cuda") else torch.float32


def _build_transcriber() -> TranscriberBundle:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    device = _transcription_device()
    dtype = _transcription_dtype(device)
    processor = AutoProcessor.from_pretrained(TRANSCRIPTION_MODEL_NAME)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        TRANSCRIPTION_MODEL_NAME,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()
    return TranscriberBundle(
        model=model,
        processor=processor,
        device=device,
        dtype=dtype,
    )


def _get_transcriber() -> TranscriberBundle:
    global _transcriber
    if _transcriber is not None:
        return _transcriber
    with _transcriber_lock:
        if _transcriber is None:
            _transcriber = _build_transcriber()
    return _transcriber
```

- [ ] **Step 4: Implement resampling and transcript generation**

Append to `voice_clone/transcription.py`:

```python
def normalize_transcript(text: str) -> str:
    return " ".join(text.split())


def _resample_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim != 1:
        raise ValueError("Transcription requires mono audio.")
    if sample_rate <= 0 or len(samples) == 0:
        raise ValueError("Transcription audio is empty or invalid.")

    waveform = torch.from_numpy(samples)
    if sample_rate != WHISPER_SAMPLE_RATE:
        waveform = audio_functional.resample(
            waveform,
            orig_freq=sample_rate,
            new_freq=WHISPER_SAMPLE_RATE,
        )
    return waveform.numpy()


def transcribe(audio: np.ndarray, sample_rate: int, language: str) -> str:
    whisper_language = SUPPORTED_LANGUAGES.get(language)
    if whisper_language is None:
        raise ValueError(f"Unsupported transcription language: {language}")

    samples = _resample_audio(audio, sample_rate)
    bundle = _get_transcriber()
    features = bundle.processor(
        samples,
        sampling_rate=WHISPER_SAMPLE_RATE,
        return_tensors="pt",
    ).input_features.to(bundle.device, dtype=bundle.dtype)

    with ACCELERATOR_LOCK, torch.inference_mode():
        generated_ids = bundle.model.generate(
            features,
            language=whisper_language,
            task="transcribe",
        )

    decoded = bundle.processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]
    return normalize_transcript(decoded)
```

- [ ] **Step 5: Add dtype and resampling tests**

Append to `tests/test_transcription.py`:

```python
def test_transcription_dtype_uses_float16_on_cuda():
    assert transcription._transcription_dtype("cuda:0") == torch.float16


def test_transcription_dtype_uses_float32_on_cpu():
    assert transcription._transcription_dtype("cpu") == torch.float32


def test_resample_audio_returns_whisper_rate_length():
    audio = np.zeros(24_000, dtype=np.float32)

    resampled = transcription._resample_audio(audio, 24_000)

    assert len(resampled) == pytest.approx(16_000, abs=2)
```

- [ ] **Step 6: Run transcription tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_transcription.py -q
```

Expected: all transcription tests pass without downloading model weights.

- [ ] **Step 7: Commit the transcription service**

```powershell
git add voice_clone\transcription.py tests\test_transcription.py
git commit -m "feat: add local whisper transcription" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>" -m "Copilot-Session: ce8fca23-6d56-42c5-be2d-4c420668e8cf"
```

---

### Task 4: Capture Orchestration and Cleanup

**Files:**
- Create: `voice_clone/capture.py`
- Create: `tests/test_capture.py`

**Interfaces:**
- Consumes:
  - `prepare_reference(source)`
  - `transcribe(audio, sample_rate, language)`
- Produces:
  - `CaptureInspection`
  - `inspect_reference(source: str | Path, language: str) -> CaptureInspection`

- [ ] **Step 1: Write failing orchestration tests**

Create `tests/test_capture.py`:

```python
from pathlib import Path

import numpy as np
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


def write_reference(path: Path):
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
```

- [ ] **Step 2: Run the tests and confirm the missing module failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_capture.py -q
```

Expected: collection fails because `voice_clone.capture` does not exist.

- [ ] **Step 3: Implement the orchestration boundary**

Create `voice_clone/capture.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
from requests import RequestException

from voice_clone.core import ReferenceAnalysis, prepare_reference
from voice_clone.transcription import transcribe


@dataclass(frozen=True)
class CaptureInspection:
    analysis: ReferenceAnalysis
    transcript: str
    transcription_error: str | None


def inspect_reference(source: str | Path, language: str) -> CaptureInspection:
    prepared = prepare_reference(source)
    try:
        audio, sample_rate = sf.read(prepared.path, dtype="float32")
        try:
            transcript = transcribe(audio, sample_rate, language)
        except (
            ImportError,
            OSError,
            RequestException,
            RuntimeError,
            ValueError,
        ) as exc:
            return CaptureInspection(
                analysis=prepared.analysis,
                transcript="",
                transcription_error=f"{type(exc).__name__}: {exc}",
            )
        return CaptureInspection(
            analysis=prepared.analysis,
            transcript=transcript,
            transcription_error=None,
        )
    finally:
        prepared.path.unlink(missing_ok=True)
```

- [ ] **Step 4: Add a blocking preparation-error regression test**

Append to `tests/test_capture.py`:

```python
def test_inspect_reference_does_not_hide_audio_validation_errors(monkeypatch):
    def fail_preparation(_):
        raise ValueError("Reference audio is silent or too quiet.")

    monkeypatch.setattr(capture, "prepare_reference", fail_preparation)

    try:
        capture.inspect_reference("source.wav", "English")
    except ValueError as exc:
        assert str(exc) == "Reference audio is silent or too quiet."
    else:
        raise AssertionError("Expected audio validation to remain blocking")


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

    try:
        capture.inspect_reference("source.wav", "English")
    except RuntimeError as exc:
        assert str(exc) == "read failed"
    else:
        raise AssertionError("Expected audio read failure")

    assert not normalized.exists()
```

- [ ] **Step 5: Run capture tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_capture.py -q
```

Expected: all capture tests pass.

- [ ] **Step 6: Commit capture orchestration**

```powershell
git add voice_clone\capture.py tests\test_capture.py
git commit -m "feat: inspect and transcribe voice captures" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>" -m "Copilot-Session: ce8fca23-6d56-42c5-be2d-4c420668e8cf"
```

---

### Task 5: Enhanced Two-Panel Gradio Experience

**Files:**
- Modify: `app.py:1-160`
- Create: `tests/test_app.py`

**Interfaces:**
- Consumes:
  - `CaptureInspection`
  - `inspect_reference(source, language)`
  - Existing `create_voice`, `synthesize`, `list_voices`, and `delete_voice`.
- Produces:
  - `format_reference_analysis(analysis) -> str`
  - `inspect_reference_ui(reference_audio, language, progress) -> tuple[str, str, str]`
  - Automatic audio-change and explicit retry events.

- [ ] **Step 1: Write failing presentation and callback tests**

Create `tests/test_app.py`:

```python
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
```

- [ ] **Step 2: Run the tests and confirm missing presentation functions**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_app.py -q
```

Expected: failures because `format_reference_analysis` and
`inspect_reference_ui` do not exist.

- [ ] **Step 3: Add capture callback and deterministic quality rendering**

Add this import to `app.py`:

```python
from voice_clone.capture import inspect_reference
from voice_clone.core import ReferenceAnalysis
```

Add these helpers before the CSS declaration:

```python
RATING_COPY = {
    "good": "Good",
    "caution": "Check",
    "poor": "Poor",
}


def _quality_card(label: str, value: str, rating: str) -> str:
    return (
        f'<div class="quality-card" data-rating="{rating}">'
        f'<span class="quality-label">{label}</span>'
        f'<strong>{value}</strong>'
        f'<span>{RATING_COPY[rating]}</span>'
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


def inspect_reference_ui(
    reference_audio,
    language,
    progress=gr.Progress(),
):
    if not reference_audio:
        return "", "", ""

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
    return result.transcript, quality, " ".join(status_parts)
```

Extend `CSS` with:

```css
.quality-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
.quality-card { border: 1px solid var(--border-color-primary); border-radius: 10px; padding: 10px; display: grid; gap: 3px; }
.quality-card[data-rating="good"] { border-color: #22c55e; }
.quality-card[data-rating="caution"] { border-color: #f59e0b; }
.quality-card[data-rating="poor"] { border-color: #ef4444; }
.quality-label { font-size: 0.8rem; opacity: 0.75; }
```

- [ ] **Step 4: Rebuild the layout as capture/setup above Type-and-Speak**

Replace the component construction inside `with gr.Blocks(...) as demo:` with
this structure while retaining the existing callbacks:

```python
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
```

Add these events before the existing `create_button.click(...)` call:

```python
reference.change(
    fn=inspect_reference_ui,
    inputs=[reference, language],
    outputs=[reference_text, quality_summary, transcription_status],
)
transcribe_again.click(
    fn=inspect_reference_ui,
    inputs=[reference, language],
    outputs=[reference_text, quality_summary, transcription_status],
)
```

Do not bind language changes directly to transcription; this preserves manual
transcript edits until the user explicitly clicks Transcribe Again.

Keep the existing responsible-use footer directly below the Type-and-Speak
workspace.

- [ ] **Step 5: Run app and regression tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_app.py tests\test_model.py tests\test_capture.py -q
```

Expected: all selected tests pass without loading model weights.

- [ ] **Step 6: Run the app import smoke check**

Run:

```powershell
.\.venv\Scripts\python.exe -c "import app; print('app_import=ok')"
```

Expected: prints `app_import=ok` without downloading Whisper or Qwen weights.

- [ ] **Step 7: Commit the two-panel experience**

```powershell
git add app.py tests\test_app.py
git commit -m "feat: add instant clone capture workflow" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>" -m "Copilot-Session: ce8fca23-6d56-42c5-be2d-4c420668e8cf"
```

---

### Task 6: Notices, Privacy Documentation, and Full Verification

**Files:**
- Modify: `THIRD_PARTY_NOTICES.md`
- Modify: `SAFETY.md`
- Modify: `CONTRIBUTING.md`

**Interfaces:**
- Consumes: Completed feature and existing `.\check.ps1`.
- Produces: Accurate model licensing, privacy behavior, developer guidance, and a green repository validation run.

- [ ] **Step 1: Update the third-party model notice**

Add this text after the Qwen entry in `THIRD_PARTY_NOTICES.md`:

```markdown
On first automatic transcription, the application also downloads
`openai/whisper-small` from Hugging Face.

- OpenAI Whisper code and model: MIT License
  - https://github.com/openai/whisper
  - https://huggingface.co/openai/whisper-small
```

- [ ] **Step 2: Document temporary audio and transcript privacy**

Add this paragraph under `## Privacy notes` in `SAFETY.md`:

```markdown
Automatic transcription runs locally after the Whisper model is cached.
Normalized reference audio is temporary and is deleted after transcription or
voice creation. Review the generated transcript before saving a voice because
the corrected reference transcript is embedded in the persisted voice profile.
```

- [ ] **Step 3: Add the manual acceptance workflow for contributors**

Append to `CONTRIBUTING.md`:

```markdown
## Manual voice-wizard verification

Before submitting changes to capture or transcription behavior:

1. Test one microphone recording and one uploaded audio file.
2. Confirm automatic transcription remains editable.
3. Confirm quiet and clipped audio display warnings.
4. Confirm a transcription failure still allows manual transcript entry.
5. Create a voice, restart the app, and synthesize with the saved profile.
6. After both models are cached, disconnect networking and repeat the workflow.

Never commit the recordings, transcripts, generated speech, or saved profiles
used for this verification.
```

- [ ] **Step 4: Run targeted tests together**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_core.py tests\test_runtime.py tests\test_transcription.py tests\test_capture.py tests\test_app.py tests\test_model.py -q
```

Expected: all tests pass and no model download begins.

- [ ] **Step 5: Run the repository verification command**

Run:

```powershell
.\check.ps1
```

Expected:

```text
No broken requirements found.
<all tests passed>
```

- [ ] **Step 6: Perform the manual acceptance pass**

Run:

```powershell
.\run.ps1
```

Verify:

1. The browser opens on `http://127.0.0.1:7860`.
2. Microphone recording auto-transcribes.
3. Uploaded WAV or compressed audio auto-transcribes through the same controls.
4. Duration, input-level, and clipping indicators render.
5. Transcript corrections are preserved unless Transcribe Again is clicked.
6. Voice creation, preview, persistence, restart, and synthesis still work.
7. With cached weights and networking disabled, transcription and synthesis work.

Stop only the process started for this verification by its recorded process ID.

- [ ] **Step 7: Commit documentation and verification guidance**

```powershell
git add THIRD_PARTY_NOTICES.md SAFETY.md CONTRIBUTING.md
git commit -m "docs: describe local transcription workflow" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>" -m "Copilot-Session: ce8fca23-6d56-42c5-be2d-4c420668e8cf"
```

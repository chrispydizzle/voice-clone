from types import SimpleNamespace

import numpy as np
import pytest
import torch

from voice_clone import model
from voice_clone import transcription


class FakeProcessor:
    def __init__(self, input_features=None):
        self.seen_audio = None
        self.seen_sample_rate = None
        self.decoded_ids = None
        self.input_features = input_features or torch.ones(1, 80, 10)

    def __call__(self, audio, sampling_rate, return_tensors):
        self.seen_audio = audio
        self.seen_sample_rate = sampling_rate
        assert return_tensors == "pt"
        return SimpleNamespace(input_features=self.input_features)

    def batch_decode(self, generated_ids, skip_special_tokens):
        self.decoded_ids = generated_ids
        assert skip_special_tokens is True
        return ["  Hello,\n   local world.  "]


class FakeModel:
    def __init__(self, events=None, lock=None):
        self.events = events
        self.lock = lock
        self.generate_kwargs = None
        self.inference_mode_enabled = None

    def generate(self, input_features, **kwargs):
        assert input_features.shape == (1, 80, 10)
        if self.events is not None:
            self.events.append("generate")
        if self.lock is not None:
            assert self.lock.active is True
        self.inference_mode_enabled = torch.is_inference_mode_enabled()
        self.generate_kwargs = kwargs
        return torch.tensor([[1, 2, 3]])


class RecordingLock:
    def __init__(self, events=None):
        self.entered = False
        self.active = False
        self.events = events

    def __enter__(self):
        self.entered = True
        self.active = True
        if self.events is not None:
            self.events.append("lock-enter")
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.events is not None:
            self.events.append("lock-exit")
        self.active = False
        return False


class FakeInputFeatures:
    def __init__(self, events, lock):
        self.events = events
        self.lock = lock

    def to(self, device, dtype):
        self.events.append("to")
        assert self.lock.active is True
        assert device == "cpu"
        assert dtype == torch.float32
        return torch.ones(1, 80, 10, dtype=dtype)


def test_normalize_transcript_collapses_whitespace():
    assert transcription.normalize_transcript("  one\n  two  ") == "one two"


def test_transcribe_uses_language_lock_and_inference_mode(monkeypatch):
    events = []
    lock = RecordingLock(events)
    processor = FakeProcessor(input_features=FakeInputFeatures(events, lock))
    model = FakeModel(events=events, lock=lock)
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
    assert model.inference_mode_enabled is True
    assert lock.entered is True
    assert events == ["lock-enter", "to", "generate", "lock-exit"]


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


def test_transcription_device_delegates_to_resolve_device(monkeypatch):
    calls = []
    monkeypatch.setattr(
        transcription,
        "resolve_device",
        lambda override_env=None: calls.append(override_env) or "cpu",
    )

    assert transcription._transcription_device() == "cpu"
    assert calls == ["VOICE_CLONE_TRANSCRIBE_DEVICE"]


def test_transcription_device_uses_transcribe_override(monkeypatch):
    monkeypatch.setenv("VOICE_CLONE_TRANSCRIBE_DEVICE", "cpu")
    monkeypatch.setenv("VOICE_CLONE_DEVICE", "cuda:7")

    assert transcription._transcription_device() == "cpu"


def test_transcription_device_falls_back_to_default_override(monkeypatch):
    monkeypatch.delenv("VOICE_CLONE_TRANSCRIBE_DEVICE", raising=False)
    monkeypatch.setenv("VOICE_CLONE_DEVICE", "cuda:7")

    assert transcription._transcription_device() == "cuda:7"


def test_transcription_dtype_uses_float16_on_cuda():
    assert transcription._transcription_dtype("cuda:0") == torch.float16


def test_transcription_dtype_uses_float32_on_cpu():
    assert transcription._transcription_dtype("cpu") == torch.float32


def test_resample_audio_returns_whisper_rate_length():
    audio = np.zeros(24_000, dtype=np.float32)

    resampled = transcription._resample_audio(audio, 24_000)

    assert len(resampled) == pytest.approx(16_000, abs=2)

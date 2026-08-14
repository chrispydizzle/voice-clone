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


def test_transcription_dtype_uses_float16_on_cuda():
    assert transcription._transcription_dtype("cuda:0") == torch.float16


def test_transcription_dtype_uses_float32_on_cpu():
    assert transcription._transcription_dtype("cpu") == torch.float32


def test_resample_audio_returns_whisper_rate_length():
    audio = np.zeros(24_000, dtype=np.float32)

    resampled = transcription._resample_audio(audio, 24_000)

    assert len(resampled) == pytest.approx(16_000, abs=2)

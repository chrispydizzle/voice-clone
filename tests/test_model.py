from dataclasses import dataclass
import sys
from types import SimpleNamespace

import numpy as np
import torch

from voice_clone import model


def test_device_selects_first_cuda_gpu(monkeypatch):
    monkeypatch.delenv("VOICE_CLONE_DEVICE", raising=False)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    assert model._device() == "cuda:0"


def test_device_honors_override(monkeypatch):
    monkeypatch.setenv("VOICE_CLONE_DEVICE", "cpu")

    assert model._device() == "cpu"


def test_dtype_uses_bfloat16_on_modern_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda _: (8, 9))

    assert model._dtype_for_device("cuda:0") == torch.bfloat16


def test_dtype_uses_float16_on_older_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda _: (6, 1))

    assert model._dtype_for_device("cuda:0") == torch.float16


def test_dtype_uses_float32_on_cpu():
    assert model._dtype_for_device("cpu") == torch.float32


def test_synthesize_requires_created_voice(tmp_path, monkeypatch):
    monkeypatch.setattr(model, "_voices", {})
    monkeypatch.setattr(model, "VOICES_DIR", tmp_path)

    try:
        model.synthesize("Hello", "missing", "English")
    except ValueError as exc:
        assert "Select a saved voice" in str(exc)
    else:
        raise AssertionError("Expected missing voice to be rejected")


@dataclass
class FakePromptItem:
    ref_code: torch.Tensor
    ref_spk_embedding: torch.Tensor
    x_vector_only_mode: bool
    icl_mode: bool
    ref_text: str


def test_saved_voice_is_listed_and_deleted(tmp_path, monkeypatch):
    monkeypatch.setattr(model, "VOICES_DIR", tmp_path)
    monkeypatch.setattr(model, "_voices", {})
    profile = model.VoiceProfile(
        prompt=[
            FakePromptItem(
                ref_code=torch.ones(2, 2),
                ref_spk_embedding=torch.ones(4),
                x_vector_only_mode=False,
                icl_mode=True,
                ref_text="Reference text.",
            )
        ],
        language="English",
    )

    model._save_voice_profile("saved_voice", profile)

    assert model.list_voices() == ["saved_voice"]
    loaded = model._load_voice_profile("saved_voice")
    assert loaded.language == "English"
    assert loaded.prompt[0].ref_text == "Reference text."
    assert torch.equal(loaded.prompt[0].ref_spk_embedding, torch.ones(4))
    assert "Deleted" in model.delete_voice("saved_voice")
    assert model.list_voices() == []


class RecordingLock:
    def __init__(self):
        self.entered = False
        self.active = False

    def __enter__(self):
        self.entered = True
        self.active = True
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.active = False
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


def test_get_model_builds_qwen_inside_shared_accelerator_lock(monkeypatch):
    lock = RecordingLock()

    class FakeQwenTTSModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            assert lock.active is True
            return "loaded-model"

    monkeypatch.setattr(model, "_model", None)
    monkeypatch.setattr(model, "_model_device", None)
    monkeypatch.setattr(model, "_ensure_sox", lambda: None)
    monkeypatch.setattr(model, "_device", lambda: "cuda:0")
    monkeypatch.setattr(model, "_dtype_for_device", lambda device: torch.float16)
    monkeypatch.setattr(model, "ACCELERATOR_LOCK", lock)
    monkeypatch.setitem(
        sys.modules,
        "qwen_tts",
        SimpleNamespace(Qwen3TTSModel=FakeQwenTTSModel),
    )

    assert model._get_model() == "loaded-model"
    assert lock.entered is True


def test_create_voice_uses_shared_lock_only_for_qwen_accelerator_work(
    tmp_path, monkeypatch
):
    lock = RecordingLock()
    normalized = tmp_path / "normalized.wav"
    output = tmp_path / "preview.wav"
    calls = []

    class FakeQwenModel:
        def create_voice_clone_prompt(self, **kwargs):
            calls.append("prompt")
            assert lock.active is True
            return ["prompt"]

        def generate_voice_clone(self, **kwargs):
            calls.append("generate")
            assert lock.active is True
            return [np.zeros(24, dtype=np.float32)], 24_000

    def write_output(*args):
        calls.append("write")
        assert lock.active is False

    def save_profile(voice_id, profile):
        calls.append("save")
        assert lock.active is False
        assert voice_id == "saved_voice"
        assert profile.language == "English"

    monkeypatch.setattr(model, "normalize_reference", lambda _: (normalized, 12.0))
    monkeypatch.setattr(model, "_output_path", lambda _: output)
    monkeypatch.setattr(model, "_get_model", lambda: FakeQwenModel())
    monkeypatch.setattr(model, "_save_voice_profile", save_profile)
    monkeypatch.setattr(model, "_voices", {})
    monkeypatch.setattr(model, "ACCELERATOR_LOCK", lock)
    monkeypatch.setattr(model.sf, "write", write_output)
    monkeypatch.setattr(model, "model_device", lambda: "cuda:0")

    audio, voice_id, status = model.create_voice(
        "reference.wav",
        "Reference text.",
        "saved_voice",
        "English",
        True,
    )

    assert audio == str(output)
    assert voice_id == "saved_voice"
    assert "12.0s" in status
    assert calls == ["prompt", "generate", "write", "save"]
    assert lock.entered is True

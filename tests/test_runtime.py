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

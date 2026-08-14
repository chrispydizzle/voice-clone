from __future__ import annotations

import os
import shutil
import threading
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import soundfile as sf

from voice_clone.core import (
    VOICE_ID_PATTERN,
    normalize_reference,
    validate_text,
    validate_voice_id,
)
from voice_clone.runtime import ACCELERATOR_LOCK, resolve_device

MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
VOICE_FORMAT_VERSION = 1
VOICES_DIR = Path("voices")
PREVIEW_TEXT = (
    "This is a preview of my cloned voice. The recording is ready for new text."
)


@dataclass(frozen=True)
class VoiceProfile:
    prompt: Any
    language: str


_model = None
_model_device: str | None = None
_model_lock = threading.Lock()
_voices: dict[str, VoiceProfile] = {}


def _device() -> str:
    return resolve_device()


def _dtype_for_device(device: str):
    import torch

    if not device.startswith("cuda"):
        return torch.float32

    index = torch.device(device).index or 0
    major, _ = torch.cuda.get_device_capability(index)
    return torch.bfloat16 if major >= 8 else torch.float16


def _ensure_sox() -> None:
    if shutil.which("sox") or os.name != "nt":
        return

    winget_root = Path(os.environ["LOCALAPPDATA"]) / "Microsoft" / "WinGet" / "Packages"
    matches = list(winget_root.glob("ChrisBagwell.SoX_*\\sox-*\\sox.exe"))
    if not matches:
        raise RuntimeError("SoX is required. Run .\\setup.ps1 to install it.")

    os.environ["PATH"] = f"{matches[0].parent}{os.pathsep}{os.environ['PATH']}"


def _get_model():
    global _model, _model_device
    if _model is not None:
        return _model

    with _model_lock:
        if _model is None:
            import torch

            _ensure_sox()
            from qwen_tts import Qwen3TTSModel

            _model_device = _device()
            with ACCELERATOR_LOCK:
                _model = Qwen3TTSModel.from_pretrained(
                    MODEL_NAME,
                    device_map=_model_device,
                    dtype=_dtype_for_device(_model_device),
                    attn_implementation="sdpa",
                )
    return _model


def model_device() -> str:
    device = _model_device or _device()
    if device.startswith("cuda"):
        import torch

        index = torch.device(device).index or 0
        return f"{device} ({torch.cuda.get_device_name(index)})"
    return device


def _output_path(prefix: str) -> Path:
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    return output_dir / f"{prefix}-{uuid.uuid4().hex[:10]}.wav"


def _voice_path(voice_id: str) -> Path:
    return VOICES_DIR / f"{validate_voice_id(voice_id)}.pt"


def list_voices() -> list[str]:
    if not VOICES_DIR.exists():
        return []
    return sorted(
        path.stem
        for path in VOICES_DIR.glob("*.pt")
        if VOICE_ID_PATTERN.fullmatch(path.stem)
    )


def _save_voice_profile(voice_id: str, profile: VoiceProfile) -> None:
    import torch

    VOICES_DIR.mkdir(exist_ok=True)
    destination = _voice_path(voice_id)
    temporary = destination.with_suffix(".tmp")
    payload = {
        "format_version": VOICE_FORMAT_VERSION,
        "language": profile.language,
        "items": [asdict(item) for item in profile.prompt],
    }
    try:
        torch.save(payload, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _load_voice_profile(voice_id: str) -> VoiceProfile:
    clean_voice_id = validate_voice_id(voice_id)
    cached = _voices.get(clean_voice_id)
    if cached is not None:
        return cached

    path = _voice_path(clean_voice_id)
    if not path.is_file():
        raise ValueError("Select a saved voice or create a new one.")

    import torch
    from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("format_version") != VOICE_FORMAT_VERSION
        or not isinstance(payload.get("language"), str)
        or not isinstance(payload.get("items"), list)
        or not payload["items"]
    ):
        raise ValueError(f"Saved voice {clean_voice_id} has an invalid format.")

    prompt = []
    for raw_item in payload["items"]:
        if not isinstance(raw_item, dict):
            raise ValueError(f"Saved voice {clean_voice_id} has an invalid prompt.")
        ref_code = raw_item.get("ref_code")
        ref_embedding = raw_item.get("ref_spk_embedding")
        if ref_code is not None and not torch.is_tensor(ref_code):
            raise ValueError(f"Saved voice {clean_voice_id} has invalid reference codes.")
        if not torch.is_tensor(ref_embedding):
            raise ValueError(f"Saved voice {clean_voice_id} has an invalid embedding.")
        prompt.append(
            VoiceClonePromptItem(
                ref_code=ref_code,
                ref_spk_embedding=ref_embedding,
                x_vector_only_mode=bool(raw_item.get("x_vector_only_mode", False)),
                icl_mode=bool(raw_item.get("icl_mode", True)),
                ref_text=raw_item.get("ref_text"),
            )
        )

    profile = VoiceProfile(prompt=prompt, language=payload["language"])
    _voices[clean_voice_id] = profile
    return profile


def delete_voice(voice_id: str) -> str:
    clean_voice_id = validate_voice_id(voice_id)
    path = _voice_path(clean_voice_id)
    if not path.is_file():
        raise ValueError(f"Saved voice {clean_voice_id} does not exist.")
    path.unlink()
    _voices.pop(clean_voice_id, None)
    return f"Deleted saved voice **{clean_voice_id}**."


def create_voice(
    reference_audio: str,
    reference_text: str,
    voice_id: str,
    language: str,
    owns_voice: bool,
) -> tuple[str, str, str]:
    if not owns_voice:
        raise ValueError("Confirm that you own this voice or have permission to clone it.")

    clean_reference_text = validate_text(reference_text)
    clean_voice_id = validate_voice_id(voice_id)
    normalized_path, duration = normalize_reference(reference_audio)
    output_path = _output_path(f"{clean_voice_id}-preview")

    try:
        model = _get_model()
        with ACCELERATOR_LOCK:
            prompt = model.create_voice_clone_prompt(
                ref_audio=str(normalized_path),
                ref_text=clean_reference_text,
                x_vector_only_mode=False,
            )
            wavs, sample_rate = model.generate_voice_clone(
                text=PREVIEW_TEXT,
                language=language,
                voice_clone_prompt=prompt,
            )
        sf.write(output_path, wavs[0], sample_rate)
        profile = VoiceProfile(prompt=prompt, language=language)
        _save_voice_profile(clean_voice_id, profile)
        _voices[clean_voice_id] = profile
    finally:
        normalized_path.unlink(missing_ok=True)

    status = (
        f"Voice **{clean_voice_id}** created from {duration:.1f}s of audio. "
        f"Running on **{model_device()}**."
    )
    return str(output_path), clean_voice_id, status


def synthesize(text: str, voice_id: str, language: str) -> tuple[str, str]:
    clean_text = validate_text(text)
    clean_voice_id = validate_voice_id(voice_id)
    profile = _load_voice_profile(clean_voice_id)
    output_path = _output_path(clean_voice_id)

    model = _get_model()
    with ACCELERATOR_LOCK:
        wavs, sample_rate = model.generate_voice_clone(
            text=clean_text,
            language=language,
            voice_clone_prompt=profile.prompt,
        )
        sf.write(output_path, wavs[0], sample_rate)

    return str(output_path), f"Generated with **{clean_voice_id}** on **{model_device()}**."

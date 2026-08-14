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

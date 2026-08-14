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

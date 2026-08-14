# Task 3 Report: Lazy Local Whisper Transcription

## Implementation

- Added `voice_clone/transcription.py` with a lazy `openai/whisper-small` loader.
- Reused `voice_clone.runtime.ACCELERATOR_LOCK` and `resolve_device("VOICE_CLONE_TRANSCRIBE_DEVICE")`.
- Applied `torch.float16` only for CUDA devices; CPU remains `torch.float32`.
- Resampled mono NumPy audio in memory to 16 kHz before Whisper feature extraction.
- Normalized decoded transcripts by collapsing whitespace.
- Rejected unsupported languages before any model work.

## Files

- `voice_clone/transcription.py`
- `tests/test_transcription.py`

## RED / GREEN Evidence

### RED

Command:

```powershell
Set-Location 'C:\Code\voice-clone\.worktrees\instant-clone-wizard'; .\.venv\Scripts\python.exe -m pytest tests\test_transcription.py -q
```

Output:

```text
=================================== ERRORS ====================================
________________ ERROR collecting tests/test_transcription.py _________________
ImportError while importing test module 'C:\Code\voice-clone\.worktrees\instant-clone-wizard\tests\test_transcription.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\Users\posad\AppData\Local\Programs\Python\Python312\Lib\importlib\__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests\test_transcription.py:8: in <module>
    from voice_clone import transcription
E   ImportError: cannot import name 'transcription' from 'voice_clone' (C:\Code\voice-clone\.worktrees\instant-clone-wizard\voice_clone\__init__.py)
=========================== short test summary info ===========================
ERROR tests/test_transcription.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 1.21s
```

### GREEN

Command:

```powershell
Set-Location 'C:\Code\voice-clone\.worktrees\instant-clone-wizard'; .\.venv\Scripts\python.exe -m pytest tests\test_transcription.py -q
```

Output:

```text
........                                                                 [100%]
8 passed in 1.26s
```

## Full Suite Result

Command:

```powershell
Set-Location 'C:\Code\voice-clone\.worktrees\instant-clone-wizard'; .\.venv\Scripts\python.exe -m pytest -q
```

Output:

```text
.............................................                            [100%]
45 passed in 4.60s
```

## Self-Review

- **Device / dtype rules:** `_transcription_device()` delegates to `resolve_device("VOICE_CLONE_TRANSCRIBE_DEVICE")`; `_transcription_dtype()` returns `float16` only for CUDA-prefixed devices and `float32` otherwise.
- **Lazy double-checked locking:** `_get_transcriber()` returns the cached bundle fast-path and initializes once under `_transcriber_lock`.
- **Shared lock identity:** `ACCELERATOR_LOCK` is imported from `voice_clone.runtime`, matching Qwen's shared accelerator lock.
- **Resampling shape/type:** `_resample_audio()` coerces to `np.float32`, requires mono 1-D input, rejects empty/invalid sample rates, and resamples in memory to 16 kHz.
- **Unsupported languages:** `transcribe()` raises `ValueError` before processor/model work when the language is not in `SUPPORTED_LANGUAGES`.
- **Processor/model API use:** `processor(..., return_tensors="pt")` feeds `.input_features`, moves features to the resolved device/dtype, and calls `model.generate(..., language=..., task="transcribe")`.
- **Import-time behavior:** the module imports `transformers` only inside `_build_transcriber()`, so importing `voice_clone.transcription` does not trigger model downloads.
- **Scope:** no UI, capture, or orchestration changes were made.

## Concerns

- No functional concerns. `_build_transcriber()` itself was intentionally not exercised in automated tests so the suite cannot trigger `from_pretrained()` downloads.

# Contributing

Contributions that improve quality, privacy, accessibility, portability, or
responsible use are welcome.

## Development setup

Use Windows 10 or 11 with Python 3.12:

```powershell
.\setup.ps1 -Dev
.\check.ps1
```

The test suite does not download model weights or require a GPU. FFmpeg is used
by the audio-normalization tests.

Set `VOICE_CLONE_DEVICE` to choose the Qwen device for cloning and synthesis.
Set `VOICE_CLONE_TRANSCRIBE_DEVICE` only when Whisper Small transcription
should use a different device; PyTorch and Torchaudio are installed by
`setup.ps1` from the selected CPU or CUDA package index and are intentionally
not listed in `requirements.txt`.

## Pull requests

1. Keep changes focused and include tests for changed behavior.
2. Run `.\check.ps1` before opening the pull request.
3. Do not commit recordings, generated audio, voice profiles, model weights, or
   transcripts containing personal information.
4. Explain any user-visible or dependency changes in the pull request.

By contributing, you agree that your contribution is licensed under the MIT
License.

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

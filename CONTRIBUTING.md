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

## Pull requests

1. Keep changes focused and include tests for changed behavior.
2. Run `.\check.ps1` before opening the pull request.
3. Do not commit recordings, generated audio, voice profiles, model weights, or
   transcripts containing personal information.
4. Explain any user-visible or dependency changes in the pull request.

By contributing, you agree that your contribution is licensed under the MIT
License.


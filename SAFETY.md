# Responsible Use

Voice cloning can enable accessibility and creative workflows, but it can also
facilitate impersonation, fraud, harassment, and misinformation.

## Use this project only when

- You are cloning your own voice, or you have explicit informed permission.
- Generated speech is disclosed as synthetic when a listener could reasonably
  mistake it for authentic speech.
- You comply with applicable privacy, publicity-rights, recording-consent, and
  election or communications laws.

## Do not use this project to

- Impersonate a person without consent.
- Bypass voice authentication or identity-verification systems.
- Commit fraud, deceive listeners, harass people, or fabricate evidence.
- Publish a voice profile, recording, or transcript without authorization.

## Privacy notes

Automatic transcription runs locally after the Whisper model is cached.
Normalized reference audio is temporary and is deleted after transcription or
voice creation. Review the generated transcript before saving a voice because
the corrected reference transcript is embedded in the persisted voice profile.
Raw microphone and upload files are owned by Gradio's cache while callbacks
run; the app requests periodic cache cleanup of files older than one hour, so
do not treat raw captures as immediately deleted.

Saved files under `voices\` contain voice embeddings and the exact reference
transcript. Treat them as sensitive biometric-like data. Generated speech is
stored under `outputs\`. Both directories are excluded from Git by default,
but they are not encrypted.

The consent checkbox is a reminder, not a technical verification mechanism.
This project does not add an audible disclosure or provenance watermark to
generated audio.

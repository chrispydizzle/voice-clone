# Third-Party Notices

This repository does not redistribute model weights. On first use, the
application downloads `Qwen/Qwen3-TTS-12Hz-1.7B-Base` from Hugging Face.

- Qwen3-TTS code and model: Apache License 2.0
  - https://github.com/QwenLM/Qwen3-TTS
  - https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base

On first automatic transcription, the application also downloads the
`openai/whisper-small` checkpoint from Hugging Face and loads it with
Transformers.

- OpenAI Whisper repository code: MIT License
  - https://github.com/openai/whisper
- Hugging Face `openai/whisper-small` checkpoint metadata: Apache License 2.0
  - https://huggingface.co/openai/whisper-small
- Transformers runtime loader: Apache License 2.0
  - https://github.com/huggingface/transformers

Python packages installed from `requirements.txt` and PyTorch's package index
remain subject to their respective licenses. The MIT License in this
repository applies only to this project's original source and documentation.

# Realtime Voice Demo

Terminal speech-to-speech demo using the OpenAI Python SDK, the Realtime API, and `gpt-realtime-2`.

## Requirements

- macOS, Linux, or another local machine with a microphone and speaker.
- Python 3.12 managed by `uv`.
- An OpenAI API key with access to `gpt-realtime-2`.
- On macOS, allow microphone access for Terminal, iTerm, or VS Code when prompted.

If installing `sounddevice` fails on macOS, install PortAudio first:

```bash
brew install portaudio
```

## Setup

```bash
uv sync
cp .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY`.

## Run

```bash
uv run realtime-voice-demo
```

The app connects immediately, starts streaming microphone audio, and plays the model's audio response through the default output device. Press `Ctrl+C` to stop.

Use headphones if possible. Open speakers can leak the assistant voice back into the microphone and accidentally interrupt the turn.

## Configuration

The demo reads these optional environment variables:

- `REALTIME_MODEL`: defaults to `gpt-realtime-2`.
- `REALTIME_VOICE`: defaults to `marin`.
- `REALTIME_REASONING_EFFORT`: defaults to `low`.
- `REALTIME_NOISE_REDUCTION`: `near_field`, `far_field`, or `off`; defaults to `near_field`.
- `REALTIME_INSTRUCTIONS`: custom behavior instructions for the assistant.

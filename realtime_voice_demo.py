from __future__ import annotations

import asyncio
import base64
import os
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from rich.console import Console


SAMPLE_RATE = 24_000
CHANNELS = 1
INPUT_CHUNK_SECONDS = 0.02
OUTPUT_BLOCK_SECONDS = 0.05

DEFAULT_INSTRUCTIONS = (
    "You are a helpful voice assistant. Reply in the same language the user uses. "
    "Keep responses concise and natural for spoken conversation."
)


@dataclass(frozen=True)
class Settings:
    model: str
    voice: str
    reasoning_effort: str
    noise_reduction: str
    instructions: str


class PCM16AudioPlayer:
    def __init__(self) -> None:
        self._queue: deque[np.ndarray[Any, np.dtype[np.int16]]] = deque()
        self._lock = threading.Lock()
        self._stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=int(SAMPLE_RATE * OUTPUT_BLOCK_SECONDS),
            callback=self._callback,
        )

    def start(self) -> None:
        self._stream.start()

    def close(self) -> None:
        self.clear()
        if self._stream.active:
            self._stream.stop()
        self._stream.close()

    def clear(self) -> None:
        with self._lock:
            self._queue.clear()

    def add(self, pcm16: bytes) -> None:
        if not pcm16:
            return

        chunk = np.frombuffer(pcm16, dtype=np.int16).copy()
        with self._lock:
            self._queue.append(chunk)

    def _callback(
        self,
        outdata: np.ndarray[Any, np.dtype[np.int16]],
        frames: int,
        _time: Any,
        _status: sd.CallbackFlags,
    ) -> None:
        output = np.zeros(frames, dtype=np.int16)
        filled = 0

        with self._lock:
            while filled < frames and self._queue:
                chunk = self._queue.popleft()
                remaining = frames - filled
                taken = min(remaining, len(chunk))
                output[filled : filled + taken] = chunk[:taken]
                filled += taken

                if len(chunk) > remaining:
                    self._queue.appendleft(chunk[remaining:])

        outdata[:, 0] = output


def load_settings() -> Settings:
    load_dotenv()

    return Settings(
        model=os.getenv("REALTIME_MODEL", "gpt-realtime-2"),
        voice=os.getenv("REALTIME_VOICE", "marin"),
        reasoning_effort=os.getenv("REALTIME_REASONING_EFFORT", "low"),
        noise_reduction=os.getenv("REALTIME_NOISE_REDUCTION", "near_field"),
        instructions=os.getenv("REALTIME_INSTRUCTIONS", DEFAULT_INSTRUCTIONS),
    )


def build_session(settings: Settings) -> dict[str, Any]:
    audio_input: dict[str, Any] = {
        "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
        "turn_detection": {
            "type": "server_vad",
            "threshold": 0.5,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 500,
            "create_response": True,
            "interrupt_response": True,
        },
    }

    noise_reduction = settings.noise_reduction.strip().lower()
    if noise_reduction in {"near_field", "far_field"}:
        audio_input["noise_reduction"] = {"type": noise_reduction}
    elif noise_reduction in {"off", "none", "false", "0"}:
        audio_input["noise_reduction"] = None
    else:
        raise ValueError(
            "REALTIME_NOISE_REDUCTION must be near_field, far_field, or off."
        )

    return {
        "type": "realtime",
        "model": settings.model,
        "instructions": settings.instructions,
        "output_modalities": ["audio"],
        "reasoning": {"effort": settings.reasoning_effort},
        "audio": {
            "input": audio_input,
            "output": {
                "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                "voice": settings.voice,
                "speed": 1.0,
            },
        },
    }


async def send_microphone_audio(connection: Any, console: Console) -> None:
    chunk_frames = int(SAMPLE_RATE * INPUT_CHUNK_SECONDS)

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        blocksize=chunk_frames,
    )
    stream.start()
    console.print("[green]Microphone is live. Speak whenever you are ready.[/green]")

    try:
        while True:
            if stream.read_available < chunk_frames:
                await asyncio.sleep(0.005)
                continue

            data, overflowed = stream.read(chunk_frames)
            if overflowed:
                console.print("[yellow]Input overflow; skipping a small audio gap.[/yellow]")

            audio = base64.b64encode(data.tobytes()).decode("ascii")
            await connection.input_audio_buffer.append(audio=audio)
            await asyncio.sleep(0)
    finally:
        stream.stop()
        stream.close()


def event_text(event: Any, name: str, default: str = "") -> str:
    value = getattr(event, name, default)
    return value if isinstance(value, str) else default


async def receive_realtime_events(
    connection: Any,
    player: PCM16AudioPlayer,
    console: Console,
) -> None:
    transcripts: dict[str, str] = {}

    async for event in connection:
        event_type = event_text(event, "type")

        if event_type == "session.created":
            session = getattr(event, "session", None)
            session_id = getattr(session, "id", "unknown")
            console.print(f"[cyan]Connected to Realtime session {session_id}.[/cyan]")
            continue

        if event_type == "session.updated":
            console.print("[cyan]Session configured.[/cyan]")
            continue

        if event_type == "input_audio_buffer.speech_started":
            player.clear()
            console.print("[dim]User speech started.[/dim]")
            continue

        if event_type == "input_audio_buffer.speech_stopped":
            console.print("[dim]User speech stopped; waiting for response.[/dim]")
            continue

        if event_type == "response.output_audio.delta":
            player.add(base64.b64decode(event.delta))
            continue

        if event_type == "response.output_audio.done":
            console.print("[dim]Assistant audio done.[/dim]")
            continue

        if event_type == "response.output_audio_transcript.delta":
            item_id = event_text(event, "item_id", "current")
            transcripts[item_id] = transcripts.get(item_id, "") + event_text(event, "delta")
            continue

        if event_type == "response.output_audio_transcript.done":
            item_id = event_text(event, "item_id", "current")
            transcript = event_text(event, "transcript") or transcripts.pop(item_id, "")
            if transcript:
                console.print(f"[bold]Assistant:[/bold] {transcript}")
            continue

        if event_type == "conversation.item.input_audio_transcription.completed":
            transcript = event_text(event, "transcript")
            if transcript:
                console.print(f"[bold]You:[/bold] {transcript}")
            continue

        if event_type == "error":
            error = getattr(event, "error", None)
            message = getattr(error, "message", "Unknown Realtime API error.")
            code = getattr(error, "code", None)
            console.print(f"[red]Realtime API error:[/red] {message} {code or ''}")
            continue


async def run() -> None:
    console = Console()
    settings = load_settings()

    if not os.getenv("OPENAI_API_KEY"):
        console.print("[red]OPENAI_API_KEY is missing.[/red] Add it to .env first.")
        raise SystemExit(1)

    session = build_session(settings)
    player = PCM16AudioPlayer()
    client = AsyncOpenAI()

    console.print(
        f"[cyan]Connecting to {settings.model} with voice {settings.voice}...[/cyan]"
    )

    try:
        player.start()
        async with client.realtime.connect(model=settings.model) as connection:
            await connection.session.update(session=session)

            receiver = asyncio.create_task(
                receive_realtime_events(connection, player, console)
            )
            sender = asyncio.create_task(send_microphone_audio(connection, console))

            done, pending = await asyncio.wait(
                {receiver, sender},
                return_when=asyncio.FIRST_EXCEPTION,
            )

            for task in pending:
                task.cancel()

            for task in done:
                task.result()
    finally:
        player.close()


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        Console().print("\n[yellow]Stopped.[/yellow]")


if __name__ == "__main__":
    main()

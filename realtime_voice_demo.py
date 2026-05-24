from __future__ import annotations

import asyncio
import base64
import html
import os
import re
import threading
import textwrap
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text


SAMPLE_RATE = 24_000
CHANNELS = 1
INPUT_CHUNK_SECONDS = 0.02
OUTPUT_BLOCK_SECONDS = 0.05
DEFAULT_GHOSTTY_FRAME_INTERVAL_SECONDS = 1 / 30
INPUT_RESUME_DELAY_SECONDS = 0.35
GHOSTTY_FRAME_DIR = (
    Path(__file__).resolve().parent / "assets" / "ghostty" / "animation_frames"
)
SPAN_RE = re.compile(r'<span class="([^"]+)">(.*?)</span>')

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
    half_duplex: bool
    ghostty_frame_interval: float


FrameLine = list[tuple[str, str]]
Frame = list[FrameLine]


@dataclass
class TranscriptEntry:
    speaker: str
    text: str
    style: str
    streaming: bool = False


def parse_ghostty_line(line: str) -> FrameLine:
    parts: FrameLine = []
    cursor = 0

    for match in SPAN_RE.finditer(line):
        if match.start() > cursor:
            parts.append(("body", html.unescape(line[cursor : match.start()])))
        parts.append((match.group(1), html.unescape(match.group(2))))
        cursor = match.end()

    if cursor < len(line):
        parts.append(("body", html.unescape(line[cursor:])))

    return parts


def load_ghostty_frames() -> list[Frame]:
    frame_paths = sorted(GHOSTTY_FRAME_DIR.glob("frame_*.txt"))
    if not frame_paths:
        raise FileNotFoundError(
            f"Ghostty animation frames are missing from {GHOSTTY_FRAME_DIR}."
        )

    return [
        [parse_ghostty_line(line) for line in path.read_text().splitlines()]
        for path in frame_paths
    ]


def frame_line_width(line: FrameLine) -> int:
    return sum(len(text) for _style, text in line)


def slice_frame_line(line: FrameLine, start: int, end: int) -> FrameLine:
    sliced: FrameLine = []
    cursor = 0

    for style, text in line:
        next_cursor = cursor + len(text)
        if next_cursor <= start:
            cursor = next_cursor
            continue
        if cursor >= end:
            break

        local_start = max(0, start - cursor)
        local_end = min(len(text), end - cursor)
        if local_start < local_end:
            sliced.append((style, text[local_start:local_end]))

        cursor = next_cursor

    return sliced


class TerminalTUI:
    def __init__(self, console: Console, settings: Settings) -> None:
        self.console = console
        self.settings = settings
        self.frames = load_ghostty_frames()
        self.frame_index = 0
        self.connected = False
        self.status = f"Connecting to {settings.model}..."
        self.entries: list[TranscriptEntry] = [
            TranscriptEntry("System", self.status, "status")
        ]
        self.assistant_entries: dict[str, TranscriptEntry] = {}
        self._lock = asyncio.Lock()
        self._live: Live | None = None
        self._animation_task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self._live = Live(
            self._render(),
            console=self.console,
            screen=True,
            auto_refresh=False,
            refresh_per_second=20,
        )
        self._live.start()
        self._refresh_unlocked()
        self._animation_task = asyncio.create_task(self._animate())

    async def stop(self) -> None:
        if self._animation_task:
            self._animation_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._animation_task

        if self._live:
            self._live.stop()
            self._live = None

    async def set_connected(self, session_id: str) -> None:
        async with self._lock:
            self.connected = True
            self.status = f"Connected to Realtime session {session_id}."
            self._append_entry_unlocked("System", self.status, "status")
            self._refresh_unlocked()

    async def set_status(self, status: str) -> None:
        async with self._lock:
            self.status = status
            self._refresh_unlocked()

    async def add_status(self, text: str) -> None:
        async with self._lock:
            self.status = text
            self._append_entry_unlocked("System", text, "status")
            self._refresh_unlocked()

    async def add_message(self, speaker: str, text: str) -> None:
        async with self._lock:
            self._append_entry_unlocked(speaker, text, speaker.lower())
            self._refresh_unlocked()

    async def append_assistant_delta(self, item_id: str, delta: str) -> None:
        if not delta:
            return

        async with self._lock:
            entry = self.assistant_entries.get(item_id)
            if entry is None:
                entry = TranscriptEntry("Assistant", "", "assistant", streaming=True)
                self.assistant_entries[item_id] = entry
                self.entries.append(entry)

            entry.text += delta
            self.status = "Assistant is speaking."
            self._trim_entries_unlocked()
            self._refresh_unlocked()

    async def finish_assistant_message(self, item_id: str, transcript: str) -> None:
        async with self._lock:
            entry = self.assistant_entries.pop(item_id, None)
            if entry:
                if transcript:
                    entry.text = transcript
                entry.streaming = False
            elif transcript:
                self._append_entry_unlocked("Assistant", transcript, "assistant")

            self.status = "Assistant audio done."
            self._refresh_unlocked()

    async def _animate(self) -> None:
        while True:
            await asyncio.sleep(self.settings.ghostty_frame_interval)
            async with self._lock:
                if self.connected:
                    self.frame_index = (self.frame_index + 1) % len(self.frames)
                self._refresh_unlocked()

    def _append_entry_unlocked(self, speaker: str, text: str, style: str) -> None:
        self.entries.append(TranscriptEntry(speaker, text, style))
        self._trim_entries_unlocked()

    def _trim_entries_unlocked(self) -> None:
        if len(self.entries) > 80:
            self.entries = self.entries[-80:]
            self.assistant_entries = {
                item_id: entry
                for item_id, entry in self.assistant_entries.items()
                if entry in self.entries
            }

    def _refresh_unlocked(self) -> None:
        if self._live:
            self._live.update(self._render(), refresh=True)

    def _render(self) -> Layout:
        width, height = self.console.size
        right_width = self._transcript_width(width)
        avatar_width = max(20, width - right_width)
        body_height = max(8, height - 3)

        layout = Layout(name="root")
        layout.split_column(Layout(name="body", ratio=1), Layout(name="footer", size=3))
        layout["body"].split_row(
            Layout(name="avatar", ratio=1),
            Layout(name="transcript", size=right_width),
        )

        layout["avatar"].update(
            Panel(
                self._render_avatar(avatar_width, body_height),
                border_style="#33415c",
                box=box.SQUARE,
            )
        )
        layout["transcript"].update(
            Panel(
                self._render_transcript(right_width, body_height),
                title="Dialogue",
                border_style="#556070",
                box=box.SQUARE,
            )
        )
        layout["footer"].update(
            Panel(self._render_status(), border_style="#33415c", box=box.SQUARE)
        )

        return layout

    def _transcript_width(self, terminal_width: int) -> int:
        if terminal_width >= 130:
            return 48
        if terminal_width >= 96:
            return 40
        return min(max(30, terminal_width // 2), max(24, terminal_width - 32))

    def _render_avatar(self, panel_width: int, panel_height: int) -> Text:
        inner_width = max(1, panel_width - 2)
        inner_height = max(1, panel_height - 2)
        frame = self.frames[self.frame_index]

        if len(frame) > inner_height:
            line_start = (len(frame) - inner_height) // 2
            visible_lines = frame[line_start : line_start + inner_height]
            top_padding = 0
        else:
            visible_lines = frame
            top_padding = (inner_height - len(frame)) // 2

        text = Text()
        for _ in range(top_padding):
            text.append("\n")

        styles = {
            "body": "#f2f5ff",
            "b": "bold #6ea8ff",
        }

        for index, line in enumerate(visible_lines):
            line_width = frame_line_width(line)
            if line_width > inner_width:
                column_start = (line_width - inner_width) // 2
                line = slice_frame_line(line, column_start, column_start + inner_width)
                left_padding = 0
            else:
                left_padding = (inner_width - line_width) // 2

            if left_padding:
                text.append(" " * left_padding)
            for style, value in line:
                text.append(value, style=styles.get(style, styles["body"]))
            if index != len(visible_lines) - 1:
                text.append("\n")

        return text

    def _render_transcript(self, panel_width: int, panel_height: int) -> Text:
        inner_width = max(16, panel_width - 4)
        inner_height = max(4, panel_height - 4)
        rendered_lines: list[Text] = []

        speaker_styles = {
            "assistant": "bold #f2f5ff",
            "you": "bold #9fe6c8",
            "status": "dim #9aa4b2",
        }
        body_styles = {
            "assistant": "#e7ecf7",
            "you": "#d8f8e8",
            "status": "dim #9aa4b2",
        }

        for entry in self.entries:
            prefix = "" if entry.style == "status" else f"{entry.speaker}: "
            style = body_styles.get(entry.style, "#e7ecf7")
            prefix_style = speaker_styles.get(entry.style, "bold")
            body = entry.text + (" |" if entry.streaming else "")
            wrapped = self._wrap_entry(prefix, body, inner_width)

            for line_index, line in enumerate(wrapped):
                line_text = Text()
                if prefix and line_index == 0:
                    line_text.append(prefix, style=prefix_style)
                elif prefix:
                    line_text.append(" " * len(prefix))
                line_text.append(line, style=style)
                rendered_lines.append(line_text)

            rendered_lines.append(Text(""))

        rendered_lines = rendered_lines[-inner_height:]
        output = Text()
        for index, line in enumerate(rendered_lines):
            output.append_text(line)
            if index != len(rendered_lines) - 1:
                output.append("\n")
        return output

    def _wrap_entry(self, prefix: str, body: str, width: int) -> list[str]:
        first_width = max(8, width - len(prefix))
        wrapped: list[str] = []
        paragraphs = body.splitlines() or [""]

        for paragraph in paragraphs:
            target_width = first_width if not wrapped else width
            lines = textwrap.wrap(
                paragraph,
                width=target_width,
                break_long_words=True,
                drop_whitespace=False,
                replace_whitespace=False,
            )
            wrapped.extend(lines or [""])

        return wrapped

    def _render_status(self) -> Text:
        text = Text()
        text.append("status ", style="dim #9aa4b2")
        text.append(self.status, style="#e7ecf7")
        text.append("   model ", style="dim #9aa4b2")
        text.append(self.settings.model, style="#bcd4ff")
        text.append("   voice ", style="dim #9aa4b2")
        text.append(self.settings.voice, style="#bcd4ff")
        return text


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

    def queued_seconds(self) -> float:
        with self._lock:
            queued_samples = sum(len(chunk) for chunk in self._queue)
        return queued_samples / SAMPLE_RATE

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


class MicrophoneGate:
    def __init__(self, active: bool) -> None:
        self.active = active
        self._enabled = asyncio.Event()
        self._enabled.set()
        self._resume_task: asyncio.Task[None] | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled.is_set()

    def pause(self) -> None:
        if not self.active:
            return
        self._cancel_resume()
        self._enabled.clear()

    def resume(self) -> None:
        self._cancel_resume()
        self._enabled.set()

    def resume_after(self, delay: float) -> None:
        if not self.active:
            return
        self._cancel_resume()
        self._resume_task = asyncio.create_task(self._resume_after(delay))

    async def close(self) -> None:
        if self._resume_task:
            self._resume_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._resume_task

    def _cancel_resume(self) -> None:
        if self._resume_task and not self._resume_task.done():
            self._resume_task.cancel()
        self._resume_task = None

    async def _resume_after(self, delay: float) -> None:
        await asyncio.sleep(delay)
        self._enabled.set()


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ValueError(f"{name} must be true or false.")


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default

    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number.") from exc

    if parsed <= 0:
        raise ValueError(f"{name} must be greater than 0.")

    return parsed


def load_settings() -> Settings:
    load_dotenv()

    return Settings(
        model=os.getenv("REALTIME_MODEL", "gpt-realtime-2"),
        voice=os.getenv("REALTIME_VOICE", "marin"),
        reasoning_effort=os.getenv("REALTIME_REASONING_EFFORT", "low"),
        noise_reduction=os.getenv("REALTIME_NOISE_REDUCTION", "near_field"),
        instructions=os.getenv("REALTIME_INSTRUCTIONS", DEFAULT_INSTRUCTIONS),
        half_duplex=env_bool("REALTIME_HALF_DUPLEX", False),
        ghostty_frame_interval=env_float(
            "GHOSTTY_FRAME_INTERVAL_SECONDS",
            DEFAULT_GHOSTTY_FRAME_INTERVAL_SECONDS,
        ),
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
            "interrupt_response": not settings.half_duplex,
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


async def send_microphone_audio(
    connection: Any,
    tui: TerminalTUI,
    microphone_gate: MicrophoneGate,
) -> None:
    chunk_frames = int(SAMPLE_RATE * INPUT_CHUNK_SECONDS)

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        blocksize=chunk_frames,
    )
    stream.start()
    await tui.add_status("Microphone is live. Speak whenever you are ready.")

    try:
        while True:
            if stream.read_available < chunk_frames:
                await asyncio.sleep(0.005)
                continue

            data, overflowed = stream.read(chunk_frames)
            if overflowed:
                await tui.add_status("Input overflow; skipping a small audio gap.")

            if not microphone_gate.enabled:
                await asyncio.sleep(0)
                continue

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
    tui: TerminalTUI,
    microphone_gate: MicrophoneGate,
) -> None:
    async for event in connection:
        event_type = event_text(event, "type")

        if event_type == "session.created":
            session = getattr(event, "session", None)
            session_id = getattr(session, "id", "unknown")
            await tui.set_connected(session_id)
            continue

        if event_type == "session.updated":
            await tui.add_status("Session configured.")
            continue

        if event_type == "input_audio_buffer.speech_started":
            player.clear()
            await tui.set_status("Listening to user speech.")
            continue

        if event_type == "input_audio_buffer.speech_stopped":
            microphone_gate.pause()
            await tui.set_status("User speech stopped; waiting for response.")
            continue

        if event_type == "response.output_audio.delta":
            microphone_gate.pause()
            player.add(base64.b64decode(event.delta))
            await tui.set_status("Assistant is speaking.")
            continue

        if event_type == "response.output_audio.done":
            resume_delay = player.queued_seconds() + INPUT_RESUME_DELAY_SECONDS
            microphone_gate.resume_after(resume_delay)
            await tui.set_status("Assistant audio done; listening resumes shortly.")
            continue

        if event_type == "response.done":
            resume_delay = player.queued_seconds() + INPUT_RESUME_DELAY_SECONDS
            microphone_gate.resume_after(resume_delay)
            await tui.set_status("Assistant done; listening resumes shortly.")
            continue

        if event_type == "response.output_audio_transcript.delta":
            item_id = event_text(event, "item_id", "current")
            await tui.append_assistant_delta(item_id, event_text(event, "delta"))
            continue

        if event_type == "response.output_audio_transcript.done":
            item_id = event_text(event, "item_id", "current")
            await tui.finish_assistant_message(item_id, event_text(event, "transcript"))
            continue

        if event_type == "conversation.item.input_audio_transcription.completed":
            transcript = event_text(event, "transcript")
            if transcript:
                await tui.add_message("You", transcript)
            continue

        if event_type == "error":
            error = getattr(event, "error", None)
            message = getattr(error, "message", "Unknown Realtime API error.")
            code = getattr(error, "code", None)
            microphone_gate.resume()
            await tui.add_status(f"Realtime API error: {message} {code or ''}".strip())
            continue


async def run() -> None:
    console = Console()
    settings = load_settings()

    if not os.getenv("OPENAI_API_KEY"):
        console.print("[red]OPENAI_API_KEY is missing.[/red] Add it to .env first.")
        raise SystemExit(1)

    session = build_session(settings)
    player = PCM16AudioPlayer()
    microphone_gate = MicrophoneGate(active=settings.half_duplex)
    client = AsyncOpenAI()
    tui = TerminalTUI(console, settings)
    tui.start()

    try:
        player.start()
        async with client.realtime.connect(model=settings.model) as connection:
            await connection.session.update(session=session)

            receiver = asyncio.create_task(
                receive_realtime_events(connection, player, tui, microphone_gate)
            )
            sender = asyncio.create_task(
                send_microphone_audio(connection, tui, microphone_gate)
            )

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
        await microphone_gate.close()
        await tui.stop()


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        Console().print("\n[yellow]Stopped.[/yellow]")


if __name__ == "__main__":
    main()

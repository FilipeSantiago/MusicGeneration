"""Streams raw RGB frames (and optional WAV audio) through ffmpeg into an mp4.

All file I/O happens in Python: frames go to ffmpeg's stdin, audio through an
inherited pipe fd, and the encoded mp4 comes back on stdout. This keeps the
writer working with sandboxed ffmpeg builds (e.g. the snap package) that
cannot see ``/tmp`` or arbitrary paths.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from pathlib import Path
from types import TracebackType
from typing import IO, Self

_CHUNK = 1 << 16


class FFmpegError(RuntimeError):
    pass


def find_ffmpeg(binary: str = "ffmpeg") -> str:
    path = shutil.which(binary)
    if path is None:
        raise FFmpegError(
            f"ffmpeg executable {binary!r} not found on PATH; install ffmpeg "
            "(e.g. `sudo apt install ffmpeg`) or set RenderConfig.ffmpeg_bin"
        )
    return path


class FFmpegWriter:
    """Context manager: ``write(frame_bytes)`` per frame, mp4 written on exit."""

    def __init__(
        self,
        out_path: Path,
        *,
        width: int,
        height: int,
        fps: int,
        audio_wav: bytes | None = None,
        duration_sec: float | None = None,
        ffmpeg_bin: str = "ffmpeg",
        crf: int = 20,
    ) -> None:
        self.out_path = Path(out_path)
        self.frame_size = width * height * 3
        self._audio_wav = audio_wav
        self._threads: list[threading.Thread] = []
        self._stderr = bytearray()
        self._errors: list[BaseException] = []
        self.frames_written = 0

        audio_read_fd: int | None = None
        audio_write_fd: int | None = None
        if audio_wav is not None:
            audio_read_fd, audio_write_fd = os.pipe()

        cmd = [
            find_ffmpeg(ffmpeg_bin),
            "-hide_banner", "-loglevel", "error", "-nostdin",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
            "-r", str(fps), "-i", "pipe:0",
        ]  # fmt: skip
        if audio_read_fd is not None:
            cmd += [
                "-f",
                "wav",
                "-i",
                f"pipe:{audio_read_fd}",
                "-map",
                "0:v",
                "-map",
                "1:a",
            ]
            cmd += ["-c:a", "aac", "-b:a", "192k"]
        cmd += [
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
        ]
        if duration_sec is not None:
            cmd += ["-t", f"{duration_sec:.3f}"]
        # Fragmented mp4 can be written to a non-seekable pipe.
        cmd += [
            "-movflags",
            "frag_keyframe+empty_moov+default_base_moof",
            "-f",
            "mp4",
            "pipe:1",
        ]

        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._out_file = self.out_path.open("wb")
        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                pass_fds=(audio_read_fd,) if audio_read_fd is not None else (),
            )
        except BaseException:
            self._out_file.close()
            for fd in (audio_read_fd, audio_write_fd):
                if fd is not None:
                    os.close(fd)
            raise
        if audio_read_fd is not None:
            os.close(audio_read_fd)

        assert self._process.stdout is not None and self._process.stderr is not None
        self._spawn(self._pump_output, self._process.stdout)
        self._spawn(self._collect_stderr, self._process.stderr)
        if audio_write_fd is not None:
            self._spawn(self._feed_audio, audio_write_fd)

    # -- background pumps ---------------------------------------------------

    def _spawn(self, target, *args) -> None:  # type: ignore[no-untyped-def]
        def run() -> None:
            try:
                target(*args)
            except Exception as exc:  # noqa: BLE001 - re-raised in close()
                self._errors.append(exc)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        self._threads.append(thread)

    def _pump_output(self, stream: IO[bytes]) -> None:
        for chunk in iter(lambda: stream.read(_CHUNK), b""):
            self._out_file.write(chunk)

    def _collect_stderr(self, stream: IO[bytes]) -> None:
        for chunk in iter(lambda: stream.read(_CHUNK), b""):
            self._stderr.extend(chunk)

    def _feed_audio(self, fd: int) -> None:
        with os.fdopen(fd, "wb") as pipe:
            try:
                pipe.write(self._audio_wav or b"")
            except BrokenPipeError:
                pass  # ffmpeg stopped reading (e.g. -t reached); not an error.

    # -- public API ---------------------------------------------------------

    def write(self, frame: bytes) -> None:
        if len(frame) != self.frame_size:
            raise ValueError(
                f"expected {self.frame_size} bytes per frame, got {len(frame)}"
            )
        assert self._process.stdin is not None
        try:
            self._process.stdin.write(frame)
        except BrokenPipeError as exc:
            self._finish()
            raise FFmpegError(self._failure_message()) from exc
        self.frames_written += 1

    def close(self) -> Path:
        returncode = self._finish()
        if returncode != 0 or self._errors:
            raise FFmpegError(self._failure_message())
        return self.out_path

    def abort(self) -> None:
        self._process.kill()
        self._finish()
        self.out_path.unlink(missing_ok=True)

    def _finish(self) -> int:
        stdin = self._process.stdin
        if stdin is not None and not stdin.closed:
            try:
                stdin.close()
            except BrokenPipeError:
                pass
        returncode = self._process.wait()
        for thread in self._threads:
            thread.join()
        if not self._out_file.closed:
            self._out_file.close()
        return returncode

    def _failure_message(self) -> str:
        detail = self._stderr.decode(errors="replace").strip()[-2000:]
        if self._errors:
            detail += f"\n{self._errors[0]!r}"
        return (
            f"ffmpeg failed (exit {self._process.returncode}): {detail or 'no output'}"
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


def remux_faststart(src: Path, dst: Path, *, ffmpeg_bin: str = "ffmpeg") -> Path:
    """Rewrite a (fragmented) mp4 as a regular one with the index up front.

    Upload pipelines (LinkedIn, X, ...) are happiest with this layout. Unlike
    :class:`FFmpegWriter`, ffmpeg must write ``dst`` itself (mp4 indexing needs a
    seekable file), so a sandboxed ffmpeg needs ``dst`` inside your home folder.
    """
    cmd = [
        find_ffmpeg(ffmpeg_bin), "-hide_banner", "-loglevel", "error", "-y",
        "-i", "pipe:0", "-c", "copy", "-movflags", "+faststart", str(dst),
    ]  # fmt: skip
    # Stream through a real pipe: sandboxed ffmpeg rejects a file fd as stdin.
    process = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    assert process.stdin is not None and process.stderr is not None
    stderr = bytearray()
    reader = threading.Thread(
        target=lambda: stderr.extend(process.stderr.read()),  # type: ignore[union-attr]
        daemon=True,
    )
    reader.start()
    try:
        with Path(src).open("rb") as source:
            shutil.copyfileobj(source, process.stdin, _CHUNK)
    except BrokenPipeError:
        pass  # ffmpeg exited early; its stderr explains why
    finally:
        try:
            process.stdin.close()
        except BrokenPipeError:
            pass
    returncode = process.wait()
    reader.join()
    if returncode != 0 or not Path(dst).is_file():
        detail = stderr.decode(errors="replace").strip()[-1000:]
        raise FFmpegError(
            f"faststart remux to {dst} failed: {detail or 'no output'} "
            "(a snap-packaged ffmpeg can only write inside your home directory)"
        )
    return Path(dst)

"""Audio streaming client utilities."""

from __future__ import annotations

import socket
import threading
from typing import Optional

import numpy as np


def ms_to_srt(ms: float) -> str:
    """Convert milliseconds to SRT timestamp."""
    ms = int(round(ms))
    s, ms = divmod(ms, 1000)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def receive_and_display(sock: socket.socket, srt_path: Optional[str] = None) -> None:
    """Receive subtitle data from the server and optionally save it as SRT."""
    buf = b""
    idx = 1
    lock = threading.Lock()
    srt_f = open(srt_path, "w", encoding="utf-8") if srt_path else None
    try:
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.decode("utf-8", "ignore").strip("\r")
                if not line:
                    continue
                parts = line.strip().split(" ", 2)
                if len(parts) >= 3:
                    try:
                        beg_ms = float(parts[0])
                        end_ms = float(parts[1])
                        text = parts[2]
                        ts = f"[{ms_to_srt(beg_ms)} → {ms_to_srt(end_ms)}]"
                        print(f"{ts} {text}")
                        if srt_f:
                            with lock:
                                srt_f.write(
                                    f"{idx}\n{ms_to_srt(beg_ms)} --> {ms_to_srt(end_ms)}\n{text}\n\n"
                                )
                                srt_f.flush()
                                idx += 1
                    except Exception:
                        print(line)
                else:
                    print(line)
    finally:
        if srt_f:
            srt_f.close()


def run_client(host: str, port: int, sr: int, block: int, srt: Optional[str]) -> None:
    """Stream system audio to the server and display subtitles."""
    import soundcard as sc  # Lazy import to avoid pulseaudio requirement during CLI parsing

    spk = sc.default_speaker()
    loop_mic = sc.get_microphone(id=str(spk.name), include_loopback=True)
    print(f"Loopback source: {spk.name}")

    print(f"Connecting to {host}:{port} ...")
    sock = socket.create_connection((host, port), timeout=5)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.settimeout(None)
    print("Connected. Streaming SYSTEM audio (Ctrl+C to stop)")

    t = threading.Thread(target=receive_and_display, args=(sock, srt), daemon=True)
    t.start()

    try:
        with loop_mic.recorder(samplerate=sr, channels=2, blocksize=block) as rec:
            while True:
                data = rec.record(numframes=block)
                if data.ndim == 2 and data.shape[1] > 1:
                    data = data.mean(axis=1)
                else:
                    data = data.reshape(-1)
                pcm16 = (np.clip(data, -1.0, 1.0) * 32767.0).astype("<i2")
                sock.sendall(pcm16.tobytes())
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        t.join()
        sock.close()


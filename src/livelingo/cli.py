#!/usr/bin/env python3
# pip install soundcard numpy
import socket, sys, threading, argparse
import soundcard as sc
import numpy as np

def ms_to_srt(ms: float) -> str:
    ms = int(round(ms))
    s, ms = divmod(ms, 1000)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def receive_and_display(sock, srt_path=None):
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
            # 서버는 라인 단위("beg_ms end_ms text")로 보낸다고 가정
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.decode("utf-8", "ignore").strip("\r")
                if not line:
                    continue
                # 파싱 시도
                parts = line.strip().split(" ", 2)
                if len(parts) >= 3:
                    try:
                        beg_ms = float(parts[0]); end_ms = float(parts[1])
                        text = parts[2]
                        ts = f"[{ms_to_srt(beg_ms)} → {ms_to_srt(end_ms)}]"
                        print(f"{ts} {text}")
                        if srt_f:
                            with lock:
                                srt_f.write(f"{idx}\n{ms_to_srt(beg_ms)} --> {ms_to_srt(end_ms)}\n{text}\n\n")
                                srt_f.flush()
                                idx += 1
                    except Exception:
                        # 형식이 다르면 원문 그대로 출력
                        print(line)
                else:
                    # 형식이 다르면 원문 그대로 출력
                    print(line)
    finally:
        if srt_f:
            srt_f.close()

def main():
    ap = argparse.ArgumentParser(description="System-audio loopback -> TCP client (prints subtitles & optional SRT)")
    ap.add_argument("host", nargs="?", default="127.0.0.1")
    ap.add_argument("port", nargs="?", type=int, default=43007)
    ap.add_argument("--sr", type=int, default=16000, help="sample rate (default: 16000)")
    ap.add_argument("--block", type=int, default=1024, help="frames per block (default: 1024=~64ms)")
    ap.add_argument("--srt", type=str, default=None, help="save subtitles to this .srt file")
    args = ap.parse_args()

    # 시스템 사운드(스피커) loopback 소스 선택
    spk = sc.default_speaker()
    loop_mic = sc.get_microphone(id=str(spk.name), include_loopback=True)
    print(f"Loopback source: {spk.name}")

    # 서버 연결
    print(f"Connecting to {args.host}:{args.port} ...")
    sock = socket.create_connection((args.host, args.port), timeout=5)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)          # ← 지연 최소화
    sock.settimeout(None) 
    print("Connected. Streaming SYSTEM audio (Ctrl+C to stop)")

    # 자막 수신/표시 스레드
    t = threading.Thread(target=receive_and_display, args=(sock, args.srt), daemon=True)
    t.start()

    # 캡처 & 전송 (공유 모드에서 WASAPI가 내부 리샘플링 가능; 장치에 따라 안 될 수 있음)
    try:
        with loop_mic.recorder(samplerate=args.sr, channels=2, blocksize=args.block) as rec:
            while True:
                # float32 [-1, 1], shape=(frames, channels)
                data = rec.record(numframes=args.block)
                # 스테레오 -> 모노 다운믹스
                if data.ndim == 2 and data.shape[1] > 1:
                    data = data.mean(axis=1)
                else:
                    data = data.reshape(-1)
                # float32 -> int16(리틀엔디언)
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

if __name__ == "__main__":
    main()
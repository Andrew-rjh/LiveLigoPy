#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streaming ASR TCP Server with VAD (RMS / WebRTC), endpointing, and delta emission.

문제 해결 포인트
- 무음에서 반복 출력 방지: VAD(webrtc 또는 RMS) + 침묵 누적 시 finish()로 세그먼트 종료
- 같은 문장 재방출 방지: 이전 누적 텍스트와의 LCP(최장 공통 접두어)를 이용해 "증분(델타)"만 전송
- 타임스탬프가 달라도 같은 문장 반복 전송 억제

필수 패키지:
  pip install numpy
선택(권장):
  pip install webrtcvad

주의:
- 클라이언트가 모노 16kHz, PCM16(LE) RAW 스트림을 보내는 전제입니다.
"""

from whisper_online import *  # add_shared_args, asr_factory, set_logging, load_audio_chunk

import sys
import argparse
import os
import logging
import numpy as np
import socket
from collections import deque

# Optional WebRTC VAD
try:
    import webrtcvad
    _HAVE_WEBRTCVAD = True
except Exception:
    _HAVE_WEBRTCVAD = False

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()

# server options
parser.add_argument("--host", type=str, default='localhost')
parser.add_argument("--port", type=int, default=43007)
parser.add_argument("--warmup-file", type=str, dest="warmup_file",
                    help="Warm up whisper with a short wav to speed up first chunk.")

# debug 억제 옵션
parser.add_argument("--no-debug", action="store_true",
                    help="Suppress DEBUG logs (force log level >= INFO).")

# VAD / endpointing 옵션 (내장 --vad 플래그와 충돌하지 않도록 --vad-mode 사용)
default_vad = 'webrtc' if _HAVE_WEBRTCVAD else 'rms'
parser.add_argument("--vad-mode", choices=["off", "rms", "webrtc"], default=default_vad,
                    help=f"VAD engine (default: {default_vad})")
parser.add_argument("--vad-rms-th", type=float, default=5e-3,
                    help="RMS VAD threshold (~-46 dBFS). Increase to be more strict.")
parser.add_argument("--vad-hang-ms", type=int, default=600,
                    help="Endpointing hang time after silence (ms).")
parser.add_argument("--webrtc-aggr", type=int, default=2,
                    help="WebRTC VAD aggressiveness 0-3 (high=more strict).")
parser.add_argument("--webrtc-hop-ms", type=int, default=20,
                    help="WebRTC VAD frame hop in ms (10/20/30 allowed).")

# options from whisper_online (여기서 이미 --vad (bool) 등 여러 공통 옵션이 추가됩니다)
add_shared_args(parser)
args = parser.parse_args()

# --no-debug 처리: whisper_online의 -l 설정보다 우선 적용
if getattr(args, "l", None) is None:
    setattr(args, "l", "INFO" if args.no_debug else "DEBUG")
elif args.no_debug and str(args.l).upper() == "DEBUG":
    setattr(args, "l", "INFO")

set_logging(args, logger, other="")

# setting whisper object by args
SAMPLING_RATE = 16000  # webrtcvad 호환 SR (8/16/32/48k 중 하나여야 함)

size = args.model
language = args.lan
asr, online = asr_factory(args)
min_chunk = args.min_chunk_size

# warm up
msg = "Whisper is not warmed up. The first chunk processing may take longer."
if args.warmup_file:
    if os.path.isfile(args.warmup_file):
        a = load_audio_chunk(args.warmup_file, 0, 1)
        try:
            asr.transcribe(a)
        except Exception as e:
            logger.warning(f"Warm up failed but continuing: {e}")
        else:
            logger.info("Whisper is warmed up.")
    else:
        logger.critical("The warm up file is not available. " + msg)
        sys.exit(1)
else:
    logger.warning(msg)

######### Server objects

import line_packet  # repo 동봉 모듈

class Connection:
    """Wraps socket conn with de-duplication of repeated texts."""
    PACKET_SIZE = 32000 * 5 * 60  # 최대 수신 바이트(클라 RAW PCM)

    def __init__(self, conn):
        self.conn = conn
        self.last_line = ""
        self.last_texts = deque(maxlen=16)  # 최근 전송 텍스트 중복 방지
        self.conn.setblocking(True)

    def send(self, line: str):
        """Avoid sending exact duplicate lines and drop repeated identical text within recent window."""
        if line == self.last_line:
            return
        parts = line.split(" ", 2)
        if len(parts) >= 3:
            text = parts[2].strip()
            # 타임스탬프가 달라도 같은 문장 반복인 경우 드롭
            if text and text in self.last_texts:
                return
            if text:
                self.last_texts.append(text)

        line_packet.send_one_line(self.conn, line)
        self.last_line = line

    def receive_lines(self):
        return line_packet.receive_lines(self.conn)

    def non_blocking_receive_audio(self):
        try:
            r = self.conn.recv(self.PACKET_SIZE)
            return r
        except ConnectionResetError:
            return None

class ServerProcessor:
    """Handle one client with VAD + endpointing + delta emission."""

    def __init__(self, c, online_asr_proc, min_chunk, vad_mode: str):
        self.connection = c
        self.online_asr_proc = online_asr_proc
        self.min_chunk = float(min_chunk)

        self.last_end = None
        self.is_first = True

        # VAD state
        self.vad_mode = vad_mode  # "off"|"rms"|"webrtc"
        self.vad_rms_th = float(args.vad_rms_th)
        self.vad_hang_ms = int(args.vad_hang_ms)
        self.sil_ms = 0

        # Delta emission state
        self.sent_text = ""       # 누적 기준 텍스트(모델 가설)
        self.last_emit_ms = 0     # 마지막 전송 타임스탬프(ms)

        # WebRTC VAD init
        if self.vad_mode == "webrtc":
            if not _HAVE_WEBRTCVAD:
                logger.warning("webrtcvad not available. Falling back to RMS VAD.")
                self.vad_mode = "rms"
            else:
                self.webrtc = webrtcvad.Vad(int(args.webrtc_aggr))
                hop = int(args.webrtc_hop_ms)
                if hop not in (10, 20, 30):
                    logger.warning("webrtc-hop-ms must be 10/20/30. Using 20.")
                    hop = 20
                self.webrtc_hop_ms = hop
                # 16kHz, 20ms frame = 320 samples = 640 bytes
                self.webrtc_frame_bytes = 2 * int(SAMPLING_RATE * self.webrtc_hop_ms / 1000)
                self.webrtc_buf = bytearray()

    # === Helpers ===

    def _decode_pcm16le(self, raw_bytes: bytes) -> np.ndarray:
        """bytes(PCM16LE mono) -> float32 [-1,1]"""
        if not raw_bytes:
            return np.array([], dtype=np.float32)
        # 길이가 홀수면 마지막 바이트를 버림(안전)
        if (len(raw_bytes) & 1) == 1:
            raw_bytes = raw_bytes[:-1]
        x = np.frombuffer(raw_bytes, dtype="<i2")  # int16 little-endian
        if x.size == 0:
            return np.array([], dtype=np.float32)
        return (x.astype(np.float32) / 32768.0).copy()

    def _rms(self, x: np.ndarray) -> float:
        if x.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(x * x)) + 1e-12)

    def _pcm16_bytes_from_float(self, a: np.ndarray) -> bytes:
        a = np.clip(a, -1.0, 1.0)
        return (a * 32767.0).astype("<i2").tobytes()

    def _webrtc_voiced_ratio(self, a: np.ndarray) -> float:
        """Return ratio of voiced frames in current buffer using webrtcvad."""
        if a.size == 0:
            return 0.0
        b = self._pcm16_bytes_from_float(a)
        self.webrtc_buf.extend(b)
        total, voiced = 0, 0
        fb = self.webrtc_frame_bytes
        while len(self.webrtc_buf) >= fb:
            chunk = self.webrtc_buf[:fb]
            del self.webrtc_buf[:fb]
            total += 1
            try:
                if self.webrtc.is_speech(bytes(chunk), SAMPLING_RATE):
                    voiced += 1
            except Exception:
                # 프레임 크기/샘플레이트 불일치 시 안전하게 무성 처리
                pass
        return (voiced / total) if total else 0.0

    def _lcp_len(self, a: str, b: str) -> int:
        """최장 공통 접두어 길이"""
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    def _norm_text(self, s: str) -> str:
        """가벼운 정규화(공백 제거 + 말미 구두점 제거)"""
        if not s:
            return s
        t = "".join(ch for ch in s if not ch.isspace())
        return t.rstrip(".,?!~…")

    # === Core ===

    def receive_audio_chunk(self):
        """
        Receive available raw pcm16le bytes and convert to float32 mono 16kHz.
        Blocks until we have at least min_chunk seconds (except after first pass).
        """
        out = []
        minlimit = int(self.min_chunk * SAMPLING_RATE)
        got_any = False
        while sum(len(x) for x in out) < minlimit:
            raw_bytes = self.connection.non_blocking_receive_audio()
            if raw_bytes is None or len(raw_bytes) == 0:
                break
            got_any = True
            audio = self._decode_pcm16le(raw_bytes)
            if audio.size:
                out.append(audio)

        if not out:
            # 첫 호출에 아직 충분치 않더라도 연결이 살아있으면 계속 대기하려면 None 대신 빈 배열을 반환할 수도 있음.
            # 여기서는 원래 동작과 동일하게 None 반환(루프에서 종료)하지 않도록, got_any가 True면 빈 배열 대신 대기 유지.
            return None if not got_any else np.array([], dtype=np.float32)

        conc = np.concatenate(out)
        if self.is_first and len(conc) < minlimit:
            # 첫 chunk가 너무 짧으면 더 받아오도록 유도
            return np.array([], dtype=np.float32)
        self.is_first = False
        return conc

    def format_output_transcript(self, o):
        # (beg_s, end_s, text) -> "beg_ms end_ms text"
        if o and o[0] is not None:
            beg, end = o[0] * 1000, o[1] * 1000
            if self.last_end is not None:
                beg = max(beg, self.last_end)
            self.last_end = end
            return "%1.0f %1.0f %s" % (beg, end, o[2])
        else:
            return None

    def _emit_delta(self, msg: str):
        """
        msg: "beg_ms end_ms text" 형태. 내부 sent_text와 비교해 델타만 전송.
        """
        parts = msg.split(" ", 2)
        if len(parts) < 3:
            return
        beg_ms = int(float(parts[0]))
        end_ms = int(float(parts[1]))
        text = parts[2]

        # 델타 계산
        lcp = self._lcp_len(self.sent_text, text)
        delta = text[lcp:].lstrip()

        # ❗ 델타가 없으면(= 동일 문장 재전송) 전송하지 않음
        if len(delta) == 0:
            return

        # 너무 잦은/짧은 갱신 억제
        if len(delta) < 2 and (end_ms - self.last_emit_ms) < 300:
            return

        # 거의 동일한 문장 반복 억제(정규화 기준)
        if self._norm_text(delta) and self._norm_text(delta) in {self._norm_text(self.sent_text)}:
            return

        # 상태 갱신
        self.sent_text = text
        self.last_emit_ms = end_ms

        # 항상 델타만 전송
        self.connection.send(f"{beg_ms} {end_ms} {delta}")

    def send_result(self, o):
        msg = self.format_output_transcript(o)
        if msg is not None:
            self._emit_delta(msg)

    def _flush_segment(self):
        """
        Prolonged silence: finalize current hypothesis if possible and reset.
        """
        if hasattr(self.online_asr_proc, "finish"):
            try:
                o = self.online_asr_proc.finish()
                self.send_result(o)
            except Exception as e:
                logger.debug(f"finish() failed or returned nothing: {e}")

        # 온라인 ASR 컨텍스트 초기화 시도 (구현에 따라 reset 또는 init 지원)
        try:
            if hasattr(self.online_asr_proc, "reset") and callable(self.online_asr_proc.reset):
                self.online_asr_proc.reset()
            elif hasattr(self.online_asr_proc, "init") and callable(self.online_asr_proc.init):
                self.online_asr_proc.init()
        except Exception as e:
            logger.debug(f"re-init/reset after flush failed: {e}")

        # reset boundaries + 누적 텍스트 상태 초기화
        self.last_end = None
        self.is_first = True
        self.sent_text = ""
        self.last_emit_ms = 0

    def _vad_gate(self, a: np.ndarray) -> bool:
        """
        Returns True if voiced (keep), False if silence (drop).
        Also maintains self.sil_ms and flush on hang.
        """
        if a.size == 0:
            # 데이터는 왔지만 길이가 0인 경우(첫 루프 등) → 유지
            return False

        if self.vad_mode == "rms":
            voiced = (self._rms(a) >= self.vad_rms_th)
        elif self.vad_mode == "webrtc":
            ratio = self._webrtc_voiced_ratio(a)
            voiced = (ratio >= 0.2)  # 경험적 기본값
        else:  # "off"
            voiced = True

        dur_ms = int(len(a) / SAMPLING_RATE * 1000)

        if voiced:
            self.sil_ms = 0
            return True
        else:
            self.sil_ms += dur_ms
            if self.sil_ms >= self.vad_hang_ms:
                self._flush_segment()
            return False

    def process(self):
        self.online_asr_proc.init()
        while True:
            a = self.receive_audio_chunk()
            if a is None:
                # 연결 종료 or 데이터 없음
                break

            if self._vad_gate(a):
                # 음성으로 판단된 경우에만 모델에 투입
                self.online_asr_proc.insert_audio_chunk(a)
                # ★중요: 전역 online이 아니라 self.online_asr_proc 사용
                o = self.online_asr_proc.process_iter()
                try:
                    self.send_result(o)
                except BrokenPipeError:
                    logger.info("broken pipe -- connection closed?")
                    break

        # 연결 종료 시 마무리(선택)
        # if hasattr(self.online_asr_proc, "finish"):
        #     o = self.online_asr_proc.finish()
        #     self.send_result(o)

# server loop
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((args.host, args.port))
    s.listen(1)
    logger.info('Listening on ' + str((args.host, args.port)))
    while True:
        conn, addr = s.accept()
        logger.info('Connected to client on {}'.format(addr))
        connection = Connection(conn)
        # 내장 공통옵션의 --vad (bool)이 꺼져 있으면 강제로 off
        has_vad_flag = getattr(args, "vad", False)
        mode = "off" if not has_vad_flag else args.vad_mode
        proc = ServerProcessor(connection, online, args.min_chunk_size, mode)
        try:
            proc.process()
        finally:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            conn.close()
            logger.info('Connection to client closed')

logger.info('Server terminating.')

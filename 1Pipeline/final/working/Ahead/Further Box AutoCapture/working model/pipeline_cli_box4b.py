"""
pipeline_cli_box4b.py  (Box4a + text-envelope autocapture)
==================================================================================
Based exactly on pipeline_cli_box3d3.py. OCR, translation, SymSpell,
paragraph grouping, TTS playback, controls, and the socket protocol are
preserved. Box4/Box4b replace only the pre-capture analysis/guidance subsystem:

  1. RELAXED EDGE MARGIN: Reduced from 3.5% to 1.5% of frame width so that
     larger text near page edges doesn't permanently trigger PAGE CUT OFF.
  2. PERCENTAGE-BASED EDGE CHECK: Page is only flagged as cut off when >15%
     of text boxes touch the edge (not just one stray box).
  3. ROTATION-SAFE PARAGRAPH SPACING: Uses line-centre pitch instead of the
     overlapping top/bottom edges of axis-aligned OCR boxes.
  4. REAL FIRST-LINE INDENT DETECTION: Confirms a paragraph boundary from the
     printed indent plus the extra paragraph spacing.
  5. HEADING SAFETY: Removes the height-only heading split that treated tilted
     body lines as headings.
  6. HEADING CLASSIFICATION: Uses geometry plus relative polygon font height;
     titles/headings remain translated and spoken but are not called paragraphs.
  7. TTS CONTROLS: A toggles pause/resume; S stops playback exactly as before.
  8. PHYSICAL PAGE GUARD: A strict paper mask plus the dominant document-text
     cluster rejects top/bottom/left/right crops instead of treating every
     scene OCR box as one page.
  9. SYNCHRONIZED STABILITY: Three distinct analyzed frames are required and
     the exact analyzed frame (not a newer unchecked frame) is captured.
 10. NON-BLOCKING GUIDANCE: Pre-rendered directional, lighting, motion, and
     focus prompts run outside the document TTS queue.
 11. PAGE-ROI QUALITY: Focus and illumination are measured on document text,
     including separate top/middle/bottom bands.
12. All non-capture behaviour from box3d3 is preserved unchanged.

Box4b changes only capture/guidance behaviour: completeness is judged from
the detected text envelope, not the physical A4 boundary; sparse and mid-row
pages use the same rule as dense pages; and stability accepts three good
observations in the latest four so one detector flicker cannot reset progress.

Usage:
    C:/Users/ali/Desktop/FYDP/fydp/Scripts/python.exe pipeline_cli_box4b.py
"""

import os
import sys
import re
import time
import json
import struct
import pickle
import queue
import threading
import socket
import cv2
import numpy as np
import torch
import ctranslate2
import transformers
import symspellpy
from symspellpy import SymSpell, Verbosity

# ══════════════════════════════════════════════════════════════════════════════
#  PATHS
# ══════════════════════════════════════════════════════════════════════════════
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_project_dir(start_dir):
    """Find the repository's Project directory from nested working copies."""
    current = os.path.abspath(start_dir)
    for _ in range(10):
        if (os.path.isdir(os.path.join(current, "1Pipeline")) and
                os.path.isdir(os.path.join(current, "pics"))):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    raise RuntimeError(f"Could not locate Project root above {start_dir}")


PROJECT_DIR  = _find_project_dir(PIPELINE_DIR)
PICS_DIR           = os.path.join(PROJECT_DIR, "pics")
CAPTURED_DIR       = os.path.join(PICS_DIR, "captured")
PARAGRAPH_TEST_DIR = os.path.join(PIPELINE_DIR, "paragraph_test_outputs")  # DEBUG — remove when done
os.makedirs(CAPTURED_DIR, exist_ok=True)
os.makedirs(PARAGRAPH_TEST_DIR, exist_ok=True)  # DEBUG — remove when done

# Persistent evidence for later capture/guidance tuning. Video encoding and
# analysis-image writes run on background threads so they do not hold up the
# receiver, OCR analysis, or the three-frame capture gate.
DEBUG_SESSION_RECORDING = True
DEBUG_VIDEO_FPS = 10.0
DEBUG_VIDEO_SIZE = (640, 360)
DEBUG_JPEG_QUALITY = 90
DEBUG_RAW_JPEG_QUALITY = 95
DEBUG_SESSION_ROOT = os.path.join(PARAGRAPH_TEST_DIR, "live_debug_sessions")

PORT = 9999


class _TeeStream:
    """Mirror stdout/stderr to the terminal and the session console log."""
    def __init__(self, original, log_file, lock):
        self._original = original
        self._log_file = log_file
        self._lock = lock

    def write(self, value):
        with self._lock:
            result = self._original.write(value)
            self._log_file.write(value)
        return result

    def flush(self):
        with self._lock:
            self._original.flush()
            self._log_file.flush()

    def __getattr__(self, name):
        return getattr(self._original, name)


class DebugSessionRecorder:
    """Record continuous video plus synchronized analysis and console evidence."""
    def __init__(self, output_root=DEBUG_SESSION_ROOT):
        os.makedirs(output_root, exist_ok=True)
        stamp = time.strftime("session_%Y%m%d_%H%M%S")
        session_dir = os.path.join(output_root, stamp)
        suffix = 1
        while os.path.exists(session_dir):
            session_dir = os.path.join(output_root, f"{stamp}_{suffix:02d}")
            suffix += 1
        os.makedirs(session_dir)
        self.session_dir = session_dir
        self.analysis_dir = os.path.join(session_dir, "analysis_frames")
        self.raw_analysis_dir = os.path.join(
            session_dir, "raw_analysis_frames")
        os.makedirs(self.analysis_dir)
        os.makedirs(self.raw_analysis_dir)

        self._started_wall = time.time()
        self._started_monotonic = time.monotonic()
        self._log_lock = threading.Lock()
        self._console_lock = threading.Lock()
        self._console_file = open(
            os.path.join(session_dir, "console.log"), "a",
            encoding="utf-8", buffering=1)
        self._metrics_file = open(
            os.path.join(session_dir, "analysis_metrics.jsonl"), "a",
            encoding="utf-8", buffering=1)
        self._events_file = open(
            os.path.join(session_dir, "events.jsonl"), "a",
            encoding="utf-8", buffering=1)
        self._video_times_file = open(
            os.path.join(session_dir, "video_timestamps.jsonl"), "a",
            encoding="utf-8", buffering=1)

        self._video_queue = queue.Queue(maxsize=8)
        self._artifact_queue = queue.Queue(maxsize=24)
        self._video_thread = threading.Thread(
            target=self._video_worker, name="debug-video-writer", daemon=True)
        self._artifact_thread = threading.Thread(
            target=self._artifact_worker, name="debug-artifact-writer", daemon=True)
        self._video_thread.start()
        self._artifact_thread.start()

        self._last_video_submit = 0.0
        self._analysis_index = 0
        self._video_written = 0
        self._video_dropped = 0
        self._analysis_dropped = 0
        self._video_path = None
        self._video_codec = None
        self._input_size = None
        self._old_stdout = None
        self._old_stderr = None
        self._closed = False
        self._write_manifest(active=True)
        self.record_event("session_started", session_dir=self.session_dir)

    @staticmethod
    def _json_safe(value):
        if isinstance(value, dict):
            return {str(k): DebugSessionRecorder._json_safe(v)
                    for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [DebugSessionRecorder._json_safe(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    def _base_record(self):
        return {
            "wall_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "elapsed_seconds": round(
                time.monotonic() - self._started_monotonic, 4),
        }

    def _write_json_line(self, handle, payload):
        line = json.dumps(self._json_safe(payload), ensure_ascii=False)
        with self._log_lock:
            handle.write(line + "\n")
            handle.flush()

    def start_console_capture(self):
        if self._old_stdout is not None:
            return
        self._old_stdout, self._old_stderr = sys.stdout, sys.stderr
        sys.stdout = _TeeStream(
            self._old_stdout, self._console_file, self._console_lock)
        sys.stderr = _TeeStream(
            self._old_stderr, self._console_file, self._console_lock)

    def record_event(self, event_type, **details):
        payload = self._base_record()
        payload["event"] = event_type
        payload.update(details)
        self._write_json_line(self._events_file, payload)

    def submit_live_frame(self, frame, sequence, frame_timestamp):
        if self._closed or frame is None:
            return
        now = time.monotonic()
        if now - self._last_video_submit < 1.0 / DEBUG_VIDEO_FPS:
            return
        self._last_video_submit = now
        if self._input_size is None:
            self._input_size = (int(frame.shape[1]), int(frame.shape[0]))
        if self._video_queue.full():
            self._video_dropped += 1
            return
        # Receiver frames are immutable after decode; queue the stable reference
        # instead of copying six megabytes on every recorded frame.
        item = (frame, int(sequence), float(frame_timestamp), time.time())
        try:
            self._video_queue.put_nowait(item)
        except queue.Full:
            self._video_dropped += 1

    def record_analysis(self, payload, assessment, state, passed,
                        stability_count, motion=None):
        self._analysis_index += 1
        name = f"analysis_{self._analysis_index:05d}.jpg"
        relative_name = os.path.join("analysis_frames", name)
        raw_relative_name = os.path.join("raw_analysis_frames", name)
        overlay = draw_quality_overlay(
            payload["frame"], assessment, stability_count)
        try:
            self._artifact_queue.put_nowait((
                os.path.join(self.raw_analysis_dir, name),
                payload["frame"], DEBUG_RAW_JPEG_QUALITY))
        except queue.Full:
            self._analysis_dropped += 1
            raw_relative_name = None
        try:
            self._artifact_queue.put_nowait((
                os.path.join(self.analysis_dir, name), overlay,
                DEBUG_JPEG_QUALITY))
        except queue.Full:
            self._analysis_dropped += 1
            relative_name = None

        outer = assessment.get("outer_box") or {}
        record = self._base_record()
        record.update({
            "analysis_index": self._analysis_index,
            "raw_frame_file": raw_relative_name,
            "overlay_file": relative_name,
            "generation": payload.get("generation"),
            "frame_sequence": payload.get("sequence"),
            "frame_timestamp": payload.get("frame_timestamp"),
            "analysis_completed_at": payload.get("completed_at"),
            "analysis_error": payload.get("error"),
            "analysis_age": assessment.get("analysis_age"),
            "analysis_fresh": assessment.get("analysis_fresh"),
            "guidance_state": state,
            "guidance_text": GUIDANCE_PROMPTS.get(state, state),
            "gate_passed": bool(passed),
            "stability_count": int(stability_count),
            "page_found": assessment.get("page_found"),
            "page_complete": assessment.get("page_complete"),
            "content_complete": assessment.get("content_complete"),
            "text_envelope_complete": assessment.get(
                "text_envelope_complete"),
            "temporal_coverage_ok": assessment.get("temporal_coverage_ok"),
            "coverage_reference_rows": assessment.get(
                "coverage_reference_rows"),
            "distance_ok": assessment.get("distance_ok"),
            "text_readable": assessment.get("text_readable"),
            "focus_ok": assessment.get("focus_ok"),
            "lighting_ok": assessment.get("lighting_ok"),
            "motion_known": assessment.get("motion_known"),
            "motion_ok": assessment.get("motion_ok"),
            "motion_score": assessment.get("motion_score"),
            "motion_details": motion or {},
            "too_far": assessment.get("too_far"),
            "too_dark": assessment.get("too_dark"),
            "glare": assessment.get("glare"),
            "missing_sides": assessment.get("missing_sides", []),
            "physical_sides": assessment.get("physical_sides", []),
            "unreadable_sides": assessment.get("unreadable_sides", []),
            "row_count": assessment.get("row_count"),
            "all_box_count": assessment.get("all_box_count"),
            "cluster_box_count": len(assessment.get("boxes", [])),
            "outer_rect": outer.get("rect"),
            "text_margins": assessment.get("text_margins"),
            "text_clearance_lines": assessment.get("text_clearance_lines"),
            "median_line_ratio": assessment.get("median_line_ratio"),
            "page_long_ratio": assessment.get("page_long_ratio"),
            "band_laps": assessment.get("band_laps"),
            "band_lap_min": assessment.get("band_lap_min"),
            "line_lap_p20": assessment.get("line_lap_p20"),
            "light_median": assessment.get("light_median"),
            "light_tile_std": assessment.get("light_tile_std"),
            "outside_text_below": assessment.get("outside_text_below"),
        })
        self._write_json_line(self._metrics_file, record)

    def _open_video_writer(self):
        for filename, codec in (
                # MJPG/AVI is recoverable by most tools after an abrupt stop;
                # MP4 can lose its final moov atom and become wholly unreadable.
                ("live_feed.avi", "MJPG"),
                ("live_feed.mp4", "mp4v")):
            path = os.path.join(self.session_dir, filename)
            writer = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc(*codec),
                DEBUG_VIDEO_FPS, DEBUG_VIDEO_SIZE)
            if writer.isOpened():
                self._video_path = path
                self._video_codec = codec
                return writer
            writer.release()
        self.record_event("video_writer_failed")
        return None

    def _video_worker(self):
        writer = None
        try:
            while True:
                item = self._video_queue.get()
                if item is None:
                    break
                frame, sequence, frame_timestamp, received_wall = item
                if writer is None:
                    writer = self._open_video_writer()
                if writer is None:
                    self._video_dropped += 1
                    continue
                resized = cv2.resize(
                    frame, DEBUG_VIDEO_SIZE, interpolation=cv2.INTER_AREA)
                writer.write(resized)
                self._video_written += 1
                if self._video_written % 100 == 0:
                    self._write_manifest(active=True)
                timing = self._base_record()
                timing.update({
                    "video_frame_index": self._video_written,
                    "source_sequence": sequence,
                    "source_frame_timestamp": frame_timestamp,
                    "received_wall_time": received_wall,
                })
                self._write_json_line(self._video_times_file, timing)
        except Exception as exc:
            self.record_event("video_writer_exception", error=str(exc))
        finally:
            if writer is not None:
                writer.release()

    def _artifact_worker(self):
        try:
            while True:
                item = self._artifact_queue.get()
                if item is None:
                    break
                path, image, jpeg_quality = item
                ok = cv2.imwrite(
                    path, image,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                if not ok:
                    self.record_event(
                        "analysis_image_write_failed", path=path)
        except Exception as exc:
            self.record_event("artifact_writer_exception", error=str(exc))

    def _write_manifest(self, active):
        manifest = {
            "script": os.path.basename(__file__),
            "session_dir": self.session_dir,
            "active": bool(active),
            "started_wall_time": self._started_wall,
            "ended_wall_time": None if active else time.time(),
            "video_target_fps": DEBUG_VIDEO_FPS,
            "video_size": list(DEBUG_VIDEO_SIZE),
            "input_size": list(self._input_size) if self._input_size else None,
            "video_file": (os.path.basename(self._video_path)
                           if self._video_path else None),
            "video_codec": self._video_codec,
            "video_frames_written": self._video_written,
            "video_frames_dropped": self._video_dropped,
            "analysis_records": self._analysis_index,
            "analysis_images_dropped": self._analysis_dropped,
            "capture_required_distinct_frames": globals().get(
                "CAPTURE_REQUIRED_DISTINCT"),
        }
        path = os.path.join(self.session_dir, "session_manifest.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)

    def close(self):
        if self._closed:
            return
        self.record_event("session_ending")
        self._closed = True
        try:
            self._video_queue.put(None, timeout=2.0)
            self._artifact_queue.put(None, timeout=2.0)
            self._video_thread.join(timeout=10.0)
            self._artifact_thread.join(timeout=10.0)
            self._write_manifest(active=False)
        finally:
            if self._old_stdout is not None:
                sys.stdout, sys.stderr = self._old_stdout, self._old_stderr
                self._old_stdout = self._old_stderr = None
            for handle in (self._metrics_file, self._events_file,
                           self._video_times_file, self._console_file):
                try:
                    handle.flush()
                    handle.close()
                except Exception:
                    pass


_DEBUG_FAILURES_REPORTED = set()


def _safe_debug_call(recorder, method_name, *args, **kwargs):
    """Keep optional evidence collection from ever stopping the pipeline."""
    if recorder is None:
        return None
    try:
        return getattr(recorder, method_name)(*args, **kwargs)
    except Exception as exc:
        failure = (method_name, type(exc).__name__, str(exc))
        if failure not in _DEBUG_FAILURES_REPORTED:
            _DEBUG_FAILURES_REPORTED.add(failure)
            print(f"[DEBUG] {method_name} failed; pipeline continues: {exc}")
        return None

# ══════════════════════════════════════════════════════════════════════════════
#  WIRE PROTOCOL — must match piweb_cli.py
# ══════════════════════════════════════════════════════════════════════════════
MSG_JSON  = 0x01
MSG_FRAME = 0x02

_send_lock = threading.Lock()
OCR_LOCK = threading.Lock()
SPELL_CORRECTOR = None


def _recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 65536))
        if not chunk:
            raise ConnectionError("Socket closed by remote")
        buf += chunk
    return buf


def send_json(sock, obj):
    data = json.dumps(obj).encode("utf-8")
    with _send_lock:
        sock.sendall(struct.pack("!BI", MSG_JSON, len(data)) + data)


def recv_msg(sock):
    type_byte = _recv_exact(sock, 1)[0]
    if type_byte == MSG_JSON:
        length = struct.unpack("!I", _recv_exact(sock, 4))[0]
        data   = _recv_exact(sock, length)
        return MSG_JSON, json.loads(data.decode("utf-8"))
    elif type_byte == MSG_FRAME:
        length = struct.unpack("!Q", _recv_exact(sock, 8))[0]
        data   = _recv_exact(sock, length)
        buf    = pickle.loads(data)
        frame  = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return MSG_FRAME, frame
    else:
        raise ValueError(f"Unknown message type: 0x{type_byte:02x}")


# ══════════════════════════════════════════════════════════════════════════════
#  AUDIO TONES  (PC speakers)
# ══════════════════════════════════════════════════════════════════════════════
def _play_tone(freq_start, freq_end=None, duration=0.15, volume=0.3):
    try:
        import sounddevice as sd
        sr = 22050
        t  = np.linspace(0, duration, int(sr * duration), False)
        if freq_end is None:
            freq_end = freq_start
        freq = np.linspace(freq_start, freq_end, len(t))
        tone = (np.sin(2 * np.pi * freq * t) * volume).astype(np.float32)
        sd.play(tone, samplerate=sr)
        sd.wait()
    except Exception:
        pass

def tone_success():
    _play_tone(400, 800, 0.2)

def tone_failure():
    _play_tone(800, 400, 0.3)

def beep_capture():
    _play_tone(1000, 1000, 0.1)

def beep_ready():
    """Short high-pitched beep signaling frame quality is good enough."""
    _play_tone(1200, 1200, 0.12, volume=0.4)


def beep_failure():
    """Distinct retry signal used when a saved frame fails post-capture QA."""
    _play_tone(700, 280, 0.32, volume=0.4)


GUIDANCE_PROMPTS = {
    "checking": "Checking text position.",
    "page_not_found": "Point your view toward the printed text.",
    "look_up": "Move your view up to include the top text.",
    "look_down": "Move your view down to include the bottom text.",
    "look_left": "Move your view left to include the left text.",
    "look_right": "Move your view right to include the right text.",
    "move_closer": "Move closer. The text is too small.",
    "move_back": "Move back slightly so all the text fits.",
    "too_dark": "The page is too dark. Improve the lighting.",
    "glare": "There is glare on the page. Change the angle.",
    "bad_lighting": "The lighting is uneven. Improve the lighting.",
    "hold_still": "Text aligned. Hold still.",
    "blurry": "The text is blurry. Adjust the distance.",
    "almost_ready": "All text is visible. Hold still.",
}

_GUIDANCE_CLIPS = {}
_GUIDANCE_CLIPS_LOCK = threading.Lock()


def prepare_guidance_clips(tts_module):
    """Synthesize short prompts once; playback never touches the TTS queue."""
    if _GUIDANCE_CLIPS:
        return
    with _GUIDANCE_CLIPS_LOCK:
        if _GUIDANCE_CLIPS:
            return
        synthesize = getattr(tts_module, "_synthesize", None)
        if synthesize is None:
            raise RuntimeError("pipertts._synthesize is unavailable")
        print("[AUDIO] Pre-synthesizing real-time guidance prompts …")
        for state, prompt in GUIDANCE_PROMPTS.items():
            chunks = synthesize(prompt)
            if not chunks:
                raise RuntimeError(f"Piper produced no audio for {prompt!r}")
            sample_rate = chunks[0].sample_rate
            channels = chunks[0].sample_channels
            arrays = []
            for chunk in chunks:
                audio = np.asarray(chunk.audio_int16_array, dtype=np.int16)
                arrays.append(
                    audio.reshape(-1, channels).astype(np.float32) / 32768.0)
            # A short lead-in wakes the DAC without delaying the instruction.
            lead_in = np.zeros(
                (int(sample_rate * 0.06), channels), dtype=np.float32)
            _GUIDANCE_CLIPS[state] = (
                np.concatenate([lead_in] + arrays, axis=0), sample_rate)
        print(f"[AUDIO] {len(_GUIDANCE_CLIPS)} spoken guidance clips ready.")


def _play_guidance_clip(state):
    import sounddevice as sd
    audio, sample_rate = _GUIDANCE_CLIPS[state]
    sd.play(audio, samplerate=sample_rate, blocking=True)


class AudioGuidance:
    """Latest-state spoken feedback, independent of the blocking TTS queue."""
    def __init__(self, tts_module, state_cooldown=4.5, global_cooldown=0.80,
                 event_callback=None):
        prepare_guidance_clips(tts_module)
        self._queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._last_by_state = {}
        self._last_any = 0.0
        self._last_state = None
        self._state_cooldown = state_cooldown
        self._global_cooldown = global_cooldown
        self._event_callback = event_callback
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _record_event(self, event_type, **details):
        if self._event_callback is None:
            return
        try:
            self._event_callback(event_type, **details)
        except Exception:
            # Debug recording must never interfere with spoken guidance.
            pass

    def notify(self, state):
        if state not in GUIDANCE_PROMPTS or self._stop_event.is_set():
            return
        now = time.monotonic()
        if now - self._last_by_state.get(state, 0.0) < self._state_cooldown:
            return
        if now - self._last_any < self._global_cooldown:
            return
        opposites = {
            "look_up": "look_down", "look_down": "look_up",
            "look_left": "look_right", "look_right": "look_left",
        }
        if (opposites.get(self._last_state) == state
                and now - self._last_any < 4.0):
            return
        directional = set(opposites)
        if (self._last_state in directional
                and state != self._last_state
                and state != "almost_ready"
                and now - self._last_any < 2.6):
            # Give the wearer time to perform the last direction before a new
            # instruction can contradict it.
            return
        self._last_by_state[state] = now
        self._last_any = now
        self._last_state = state
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(state)
            print(f"  [GUIDANCE] {GUIDANCE_PROMPTS[state]}")
            self._record_event(
                "guidance_queued", state=state,
                prompt=GUIDANCE_PROMPTS[state])
        except queue.Full:
            pass

    def _worker(self):
        while not self._stop_event.is_set():
            try:
                state = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if state is None:
                break
            try:
                self._record_event(
                    "guidance_started", state=state,
                    prompt=GUIDANCE_PROMPTS[state])
                _play_guidance_clip(state)
                self._record_event(
                    "guidance_finished", state=state,
                    prompt=GUIDANCE_PROMPTS[state])
            except Exception as exc:
                print(f"  [GUIDANCE] Playback failed: {exc}")
                self._record_event(
                    "guidance_failed", state=state, error=str(exc))

    def stop(self):
        self._stop_event.set()
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=0.5)


# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
PREPROCESS_CONFIG = {
    # Winner of the 45-image / 225-prediction empirical sweep.
    "variant": "V_A_RAW_GRAYSCALE",
}


def preprocess_image(img_path: str):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Could not read image: {img_path}")
        return None
    # V_A: deliberately no blur, sharpening, CLAHE, gamma, or resizing.
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # [V6 FIX] GaussianBlur removed — C525 plastic lens is already soft;
    # blurring on top of it merges thin character strokes and kills OCR on 11-12pt text.
    # gray     = cv2.GaussianBlur(gray, (3, 3), 0)


def unsharp_mask(image, sigma=1.0, strength=1.5):
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)


# ══════════════════════════════════════════════════════════════════════════════
#  NLLB 1.3B  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
if torch.cuda.is_available():
    NLLB_DEVICE       = "cuda"
    NLLB_COMPUTE_TYPE = "float16"
    os.environ["ARGOS_DEVICE_TYPE"] = "cuda"
    print("[INFO] Hardware Target: NVIDIA GPU (CUDA)")
else:
    NLLB_DEVICE       = "cpu"
    NLLB_COMPUTE_TYPE = "int8"
    print("[INFO] Hardware Target: CPU ONLY (Int8)")

NLLB_MODEL_DIR  = os.path.join(PROJECT_DIR, "nllb-200-1.3B-ct2")
NLLB_BASE_MODEL = "facebook/nllb-200-distilled-1.3B"

NLLB_LANG_MAP = {
    "zh": "zho_Hans",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "en": "eng_Latn",
}

PIPELINE_LANG_TO_NLLB = {
    "french":  "fr",
    "chinese": "zh",
    "spanish": "es",
}

# PaddleOCR's language codes differ from NLLB's. PP-OCRv6 medium is a
# multilingual model, but this audited mapping selects the correct script.
PADDLE_OCR_LANG_MAP = {
    "french": "en",
    "chinese": "ch",
    "spanish": "en",
    "english": "en",
}


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADERS
# ══════════════════════════════════════════════════════════════════════════════
def load_ocr_engine(language):
    from paddleocr import PaddleOCR
    ocr_device = "gpu" if NLLB_DEVICE == "cuda" else "cpu"
    ocr_lang = PADDLE_OCR_LANG_MAP[language]
    print(f"[LOAD] OCR language audit: {language} -> PaddleOCR lang='{ocr_lang}'")
    print(f"[LOAD] Initializing PaddleOCR (onnxruntime, {ocr_device}) …")
    t = time.time()
    ocr = PaddleOCR(
        lang=ocr_lang,
        ocr_version="PP-OCRv6",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        engine="onnxruntime",
        device=ocr_device,
    )
    print(f"[LOAD] PaddleOCR ready in {time.time()-t:.2f}s")
    return ocr


def load_unwarper():
    from paddleocr import TextImageUnwarping
    print("[LOAD] Initializing UVDoc unwarper …")
    t = time.time()
    unwarper = TextImageUnwarping(model_name="UVDoc", engine="paddle")
    print(f"[LOAD] UVDoc ready in {time.time()-t:.2f}s")
    return unwarper


def load_nllb_translator(language):
    """Load NLLB-200 1.3B for the given language (skip if English)."""
    if language == "english":
        return None, None
    print("[LOAD] Loading NLLB-200 1.3B Distilled …")
    t = time.time()
    if not os.path.exists(NLLB_MODEL_DIR):
        print(f"[INFO] Compiling {NLLB_BASE_MODEL} → CTranslate2 …")
        cmd = (
            f"ct2-transformers-converter --model {NLLB_BASE_MODEL} "
            f"--output_dir {NLLB_MODEL_DIR} --force --quantization {NLLB_COMPUTE_TYPE}"
        )
        if os.system(cmd) != 0:
            raise RuntimeError("[ERROR] Model compilation failed.")
    translator = ctranslate2.Translator(NLLB_MODEL_DIR, device=NLLB_DEVICE, compute_type=NLLB_COMPUTE_TYPE)
    tokenizer  = transformers.AutoTokenizer.from_pretrained(NLLB_BASE_MODEL)
    print(f"[LOAD] NLLB 1.3B ready in {time.time()-t:.2f}s")
    return translator, tokenizer


def load_tts():
    if PIPELINE_DIR not in sys.path:
        sys.path.insert(0, PIPELINE_DIR)
    import pipertts
    print("[LOAD] Pre-loading Piper TTS voice …")
    t = time.time()
    pipertts.ensure_model()
    pipertts.preload()
    prepare_guidance_clips(pipertts)
    print(f"[LOAD] TTS ready in {time.time()-t:.2f}s")
    return pipertts


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE STAGES
# ══════════════════════════════════════════════════════════════════════════════
def save_frame(frame, run_count):
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename  = f"capture_cli_{run_count:03d}_{timestamp}.jpg"
    filepath  = os.path.join(CAPTURED_DIR, filename)
    # Save as high-quality JPEG (95%) instead of slow PNG
    cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  💾 Saved → {filepath}")
    return filepath


def _extract_ocr_lines(result, image_shape):
    """Return PaddleOCR text, confidence, and matching boxes as line records."""
    image_h, image_w = image_shape[:2]
    records = []

    for res in result:
        texts = list(res.get("rec_texts", []))
        scores = list(res.get("rec_scores", []))
        boxes = res.get("rec_boxes")
        polygons = res.get("rec_polys", res.get("dt_polys", []))

        for index, raw_text in enumerate(texts):
            text = str(raw_text).strip()
            if not text:
                continue

            bbox = None
            polygon = None
            if polygons is not None and index < len(polygons):
                candidate_polygon = np.asarray(
                    polygons[index], dtype=np.float32).reshape(-1, 2)
                if candidate_polygon.size:
                    polygon = candidate_polygon
            if boxes is not None and index < len(boxes):
                box = np.asarray(boxes[index]).reshape(-1)
                if box.size >= 4:
                    bbox = tuple(int(round(value)) for value in box[:4])
            if bbox is None and polygon is not None:
                if polygon.size:
                    bbox = (
                        int(round(np.min(polygon[:, 0]))),
                        int(round(np.min(polygon[:, 1]))),
                        int(round(np.max(polygon[:, 0]))),
                        int(round(np.max(polygon[:, 1]))),
                    )
            if bbox is None:
                # Coordinate-free fallback keeps the original OCR flow usable.
                y1 = index * 24
                bbox = (0, y1, max(1, image_w), y1 + 20)

            x1, y1, x2, y2 = bbox
            x1 = max(0, min(image_w - 1, x1))
            y1 = max(0, min(image_h - 1, y1))
            x2 = max(x1 + 1, min(image_w, x2))
            y2 = max(y1 + 1, min(image_h, y2))
            font_height = float(y2 - y1)
            if polygon is not None and len(polygon) >= 3:
                _, rectangle_size, _ = cv2.minAreaRect(polygon)
                polygon_sides = [
                    float(side) for side in rectangle_size if side > 0]
                if polygon_sides:
                    font_height = min(polygon_sides)
            score = float(scores[index]) if index < len(scores) else 0.0
            records.append({
                "text": text,
                "score": score,
                "bbox": (x1, y1, x2, y2),
                "width": x2 - x1,
                "height": y2 - y1,
                "font_height": max(1.0, font_height),
                "center_x": (x1 + x2) / 2.0,
                "center_y": (y1 + y2) / 2.0,
            })

    return records


def merge_ocr_fragments_into_lines(records):
    """Merge PaddleOCR fragments that occupy the same printed text row."""
    if not records:
        return []

    median_height = max(
        8.0, float(np.median([record["height"] for record in records])))
    ordered = sorted(records, key=lambda record: (
        record["center_y"], record["bbox"][0]))
    rows = []

    for record in ordered:
        selected_row = None
        for row in reversed(rows[-4:]):
            center_difference = abs(record["center_y"] - row["center_y"])
            if center_difference > 0.30 * median_height:
                continue
            rx1, _, rx2, _ = record["bbox"]
            bx1, _, bx2, _ = row["bbox"]
            horizontal_gap = max(0, max(rx1, bx1) - min(rx2, bx2))
            if horizontal_gap <= 6.0 * median_height:
                selected_row = row
                break

        if selected_row is None:
            rows.append({
                "fragments": [record],
                "bbox": record["bbox"],
                "center_y": record["center_y"],
            })
            continue

        selected_row["fragments"].append(record)
        fragments = selected_row["fragments"]
        x1 = min(fragment["bbox"][0] for fragment in fragments)
        y1 = min(fragment["bbox"][1] for fragment in fragments)
        x2 = max(fragment["bbox"][2] for fragment in fragments)
        y2 = max(fragment["bbox"][3] for fragment in fragments)
        selected_row["bbox"] = (x1, y1, x2, y2)
        selected_row["center_y"] = float(np.mean([
            fragment["center_y"] for fragment in fragments]))

    merged = []
    for row in rows:
        fragments = sorted(
            row["fragments"], key=lambda fragment: fragment["bbox"][0])
        x1, y1, x2, y2 = row["bbox"]
        merged.append({
            "text": " ".join(fragment["text"] for fragment in fragments),
            "score": float(np.mean([
                fragment["score"] for fragment in fragments])),
            "bbox": (x1, y1, x2, y2),
            "width": x2 - x1,
            "height": y2 - y1,
            "font_height": float(np.median([
                fragment.get("font_height", fragment["height"])
                for fragment in fragments])),
            "center_x": (x1 + x2) / 2.0,
            "center_y": (y1 + y2) / 2.0,
        })
    return sorted(merged, key=lambda line: (
        line["center_y"], line["bbox"][0]))


def _horizontal_overlap_ratio(first, second):
    left = max(first[0], second[0])
    right = min(first[2], second[2])
    overlap = max(0, right - left)
    smaller_width = max(1, min(first[2] - first[0], second[2] - second[0]))
    return overlap / smaller_width


def group_ocr_lines_into_paragraphs(lines, image_shape):
    """
    Group OCR lines into printed paragraphs using rotation-safe geometry.

    PaddleOCR returns axis-aligned rectangles. On a tilted or unwarped page,
    neighbouring rectangles often overlap vertically even where the document
    contains a clear blank line. Paragraph spacing is therefore measured from
    line centre to line centre. A boundary is confirmed by the first-line
    indent deliberately present in the test documents; punctuation remains
    supporting evidence because OCR can drop full stops.
    """
    if not lines:
        return []

    ordered = sorted(lines, key=lambda line: (
        line["center_y"], line["bbox"][0]))
    heights = [line["height"] for line in ordered if line["height"] > 0]
    widths = [line["width"] for line in ordered if line["width"] > 0]
    median_height = max(
        8.0, float(np.median(heights)) if heights else 20.0)
    median_width = max(
        40.0, float(np.median(widths)) if widths else 200.0)

    # Estimate normal baseline-to-baseline spacing from the lower 70% of
    # plausible pitches. Paragraph gaps and missed OCR rows occupy the upper
    # tail, so neither can inflate ordinary line spacing.
    plausible_pitches = []
    pitch_floor = max(6.0, 0.22 * median_height)
    for previous, current in zip(ordered, ordered[1:]):
        pitch = current["center_y"] - previous["center_y"]
        overlap = _horizontal_overlap_ratio(
            previous["bbox"], current["bbox"])
        centre_delta = abs(previous["center_x"] - current["center_x"])
        if (pitch >= pitch_floor and
                (overlap >= 0.15 or
                 centre_delta <= 0.35 * median_width)):
            plausible_pitches.append(float(pitch))

    if plausible_pitches:
        plausible_pitches.sort()
        lower_count = max(1, int(np.ceil(0.70 * len(plausible_pitches))))
        typical_pitch = float(np.median(
            plausible_pitches[:lower_count]))
    else:
        typical_pitch = 0.70 * median_height
    typical_pitch = max(6.0, typical_pitch)

    body_like = [
        line for line in ordered if line["width"] >= 0.55 * median_width]
    if not body_like:
        body_like = ordered
    text_left = float(np.median([
        line["bbox"][0] for line in body_like]))
    text_right = float(np.median([
        line["bbox"][2] for line in body_like]))
    text_centre = (text_left + text_right) / 2.0
    centre_tolerance = max(24.0, 0.11 * median_width)
    indent_threshold = max(12.0, 0.60 * typical_pitch)
    strong_indent_threshold = max(16.0, 0.85 * typical_pitch)
    body_font_candidates = [
        line.get("font_height", line["height"])
        for line in body_like
        if line.get("font_height", line["height"]) > 0]
    page_body_font_height = max(
        4.0,
        float(np.median(body_font_candidates))
        if body_font_candidates else median_height)

    def ends_sentence(text):
        return bool(re.search(
            r"[.!?。！？；;:][\"'”’\)\]]*$", text.strip()))

    def begins_lowercase(text):
        match = re.search(r"[A-Za-z]", text)
        return bool(match and match.group(0).islower())

    def title_case_ratio(text):
        words = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
        if not words:
            return 0.0
        return sum(word[0].isupper() for word in words) / len(words)

    def local_body_font_height(index):
        nearby = []
        for other_index, other in enumerate(ordered):
            if other_index == index:
                continue
            if other["width"] < 0.48 * median_width:
                continue
            candidate_height = other.get("font_height", other["height"])
            if candidate_height <= 0:
                continue
            nearby.append((abs(other_index - index), candidate_height))
        nearby.sort(key=lambda item: item[0])
        local_values = [height for _, height in nearby[:6]]
        if local_values:
            return max(4.0, float(np.median(local_values)))
        return page_body_font_height

    def explicit_heading(line):
        text = line["text"].strip()
        return bool(
            re.match(
                r"(?i)^(?:[\[\(\|]?\s*[0-9oO]{1,3}\s*[\]\)\|]?\s*)?"
                r"chapter\s+[ivxlcdm0-9]+", text)
            or re.match(r"^\d+\.\d+\s+\S", text)
            or re.match(
                r"(?i)^(?:section|appendix)\s+[A-Z0-9IVXLC]+", text)
        )

    def starts_new_paragraph(line):
        text = line["text"].strip()
        return bool(
            re.match(
                r"^[\[\(\|]\s*[0-9oO]{1,3}\s*[\]\)\|]", text)
            or re.match(r"^[0-9oO]{1,3}\]", text)
            or re.match(
                r"^(?:[•●▪◦*-]|\d+[.)])\s+\S", text)
            or re.match(r"^\d+\.\d+\s+\S", text)
            or re.match(
                r"(?i)^chapter\s+[ivxlcdm0-9]+", text)
        )

    def pitch_before(index):
        if index <= 0:
            return 0.0
        return (
            ordered[index]["center_y"] -
            ordered[index - 1]["center_y"])

    def pitch_after(index):
        if index + 1 >= len(ordered):
            return 0.0
        return (
            ordered[index + 1]["center_y"] -
            ordered[index]["center_y"])

    def classify_visual_heading(index):
        line = ordered[index]
        text = line["text"].strip()
        local_font_height = local_body_font_height(index)
        font_scale = (
            line.get("font_height", line["height"]) /
            max(1.0, local_font_height))
        line["font_scale"] = float(font_scale)
        line["local_body_font_height"] = float(local_font_height)
        if explicit_heading(line):
            return True, "explicit_heading_pattern"
        if not text or len(text) > 120:
            return False, "ordinary_text"

        # A colon may legitimately finish a heading. Full sentence endings are
        # the stronger body-text cue that rejects a heading classification.
        full_sentence_ending = (
            ends_sentence(text) and
            not text.rstrip().endswith((":", ";", "：", "；")))
        if full_sentence_ending:
            return False, "sentence_ending"
        centred = (
            abs(line["center_x"] - text_centre) <= centre_tolerance)
        shortish = line["width"] <= 0.88 * median_width
        nearby_gap = max(pitch_before(index), pitch_after(index))
        title_like = title_case_ratio(text) >= 0.55
        title_words = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
        block_overlap = max(
            0.0,
            min(line["bbox"][2], text_right) -
            max(line["bbox"][0], text_left))
        in_text_column = (
            block_overlap / max(1.0, min(line["width"], text_right - text_left))
            >= 0.45)

        # The first line is commonly a title. Title Case also handles the
        # left-aligned titles in multilingual sheets. Requiring the dominant
        # text column prevents a short background word becoming the title.
        if (index == 0 and in_text_column and
                pitch_after(index) >= 1.15 * typical_pitch and (
                    (centred and shortish) or
                    (title_like and len(title_words) >= 2))):
            return True, "top_title_geometry"

        if (in_text_column and centred and shortish and
                nearby_gap >= 1.25 * typical_pitch):
            return True, "centred_isolated_heading"

        # 20 pt over 16 pt is nominally 1.25x; 22 pt is 1.375x. Slightly
        # tolerant thresholds absorb OCR polygon measurement noise. Font size
        # is never sufficient by itself: layout evidence is still mandatory.
        if (in_text_column and font_scale >= 1.20 and shortish and (
                centred or title_like or
                nearby_gap >= 1.10 * typical_pitch)):
            return True, "large_font_plus_layout"
        if (in_text_column and font_scale >= 1.32 and
                (shortish or title_like) and
                nearby_gap >= 1.05 * typical_pitch):
            return True, "very_large_font_plus_spacing"
        return False, "ordinary_text"

    def continuation_left(index):
        # A first line is indented relative to following continuation lines.
        # Looking ahead avoids page-edge drift caused by perspective.
        following = ordered[index + 1:min(len(ordered), index + 3)]
        if following:
            return float(np.median([
                line["bbox"][0] for line in following]))
        preceding = ordered[max(0, index - 3):index]
        if preceding:
            return float(np.median([
                line["bbox"][0] for line in preceding]))
        return float(ordered[index]["bbox"][0])

    heading_results = [
        classify_visual_heading(index) for index in range(len(ordered))]
    heading_flags = [result[0] for result in heading_results]
    for line, (is_heading, evidence) in zip(ordered, heading_results):
        line["is_heading"] = bool(is_heading)
        line["heading_evidence"] = evidence
    paragraph_lines = [[ordered[0]]]
    paragraph_reasons = ["start_of_page"]

    ordered[0]["paragraph_pitch"] = 0.0
    ordered[0]["paragraph_gap_ratio"] = 0.0
    ordered[0]["paragraph_indent_delta"] = 0.0
    ordered[0]["paragraph_break_reason"] = "start_of_page"

    for index in range(1, len(ordered)):
        previous = ordered[index - 1]
        current = ordered[index]
        previous_box = previous["bbox"]
        current_box = current["bbox"]
        pitch = current["center_y"] - previous["center_y"]
        gap_ratio = pitch / typical_pitch
        overlap = _horizontal_overlap_ratio(previous_box, current_box)
        centre_delta = abs(
            previous["center_x"] - current["center_x"])
        same_column = (
            overlap >= 0.15 or
            centre_delta <= 0.35 * median_width)

        indent_delta = current_box[0] - continuation_left(index)
        indented_start = indent_delta >= indent_threshold
        strong_indented_start = indent_delta >= strong_indent_threshold
        previous_terminal = ends_sentence(previous["text"])
        previous_short = previous["width"] <= 0.72 * median_width
        lowercase_continuation = begins_lowercase(current["text"])

        reason = None
        if not same_column:
            reason = "column_change"
        elif (heading_flags[index] and heading_flags[index - 1] and
                gap_ratio <= 1.35):
            reason = None
        elif heading_flags[index]:
            reason = "heading"
        elif heading_flags[index - 1]:
            reason = "after_heading"
        elif starts_new_paragraph(current):
            reason = "numbered_or_bulleted_start"
        elif (indented_start and gap_ratio >= 1.16) or (
                strong_indented_start and gap_ratio >= 1.04):
            reason = "first_line_indent_plus_spacing"
        elif (gap_ratio >= 1.70 and previous_short and
                not lowercase_continuation):
            reason = "blank_gap_after_short_line"
        elif (gap_ratio >= 2.20 and previous_terminal and
                not lowercase_continuation):
            reason = "large_gap_after_sentence"

        # A missed OCR row can also create a large centre pitch. Without an
        # indent, heading, short final row, or sentence boundary, keep the
        # lines together instead of inventing a paragraph.
        current["paragraph_pitch"] = float(pitch)
        current["paragraph_gap_ratio"] = float(gap_ratio)
        current["paragraph_indent_delta"] = float(indent_delta)
        current["paragraph_break_reason"] = reason or "continuation"

        if reason is None:
            paragraph_lines[-1].append(current)
        else:
            paragraph_lines.append([current])
            paragraph_reasons.append(reason)

    image_h, image_w = image_shape[:2]
    padding = max(4, int(round(0.18 * typical_pitch)))
    paragraphs = []
    for number, (grouped_lines, reason) in enumerate(
            zip(paragraph_lines, paragraph_reasons), start=1):
        x1 = max(
            0, min(line["bbox"][0] for line in grouped_lines) - padding)
        y1 = max(
            0, min(line["bbox"][1] for line in grouped_lines) - padding)
        x2 = min(
            image_w,
            max(line["bbox"][2] for line in grouped_lines) + padding)
        y2 = min(
            image_h,
            max(line["bbox"][3] for line in grouped_lines) + padding)
        confidence_values = [line["score"] for line in grouped_lines]
        paragraphs.append({
            "number": number,
            "region_id": number,
            "source_text": " ".join(
                line["text"] for line in grouped_lines),
            "bbox": (x1, y1, x2, y2),
            "lines": grouped_lines,
            "confidence": (
                float(np.mean(confidence_values))
                if confidence_values else 0.0),
            "break_reason": reason,
            "typical_line_pitch": typical_pitch,
        })

    body_number = 0
    heading_number = 0
    page_title_assigned = False
    for region in paragraphs:
        first_line = region["lines"][0]
        first_text = first_line["text"].strip()
        heading_region = bool(first_line.get("is_heading", False))
        centre_y = float(np.mean([
            line["center_y"] for line in region["lines"]]))
        explicit_section_marker = bool(re.match(
            r"(?i)^(?:chapter|section|appendix)\b|^\d+\.\d+\s+\S",
            first_text))

        if (heading_region and not page_title_assigned and
                centre_y <= 0.30 * image_h and
                not explicit_section_marker):
            region_type = "page_title"
            label = "TITLE"
            paragraph_number = None
            assigned_heading_number = None
            page_title_assigned = True
        elif heading_region:
            heading_number += 1
            region_type = "section_heading"
            label = f"H{heading_number}"
            paragraph_number = None
            assigned_heading_number = heading_number
        else:
            body_number += 1
            region_type = "paragraph"
            label = f"P{body_number}"
            paragraph_number = body_number
            assigned_heading_number = None

        region["region_type"] = region_type
        region["label"] = label
        region["paragraph_number"] = paragraph_number
        region["heading_number"] = assigned_heading_number
        region["font_scale"] = max(
            line.get("font_scale", 1.0) for line in region["lines"])
        region["heading_evidence"] = first_line.get(
            "heading_evidence", "ordinary_text")

    # Keep adjacent debug/preview rectangles separate. The boundary is the
    # midpoint between the final baseline of one paragraph and the first
    # baseline of the next, so padding cannot cover the printed blank row.
    for previous, current in zip(paragraphs, paragraphs[1:]):
        boundary = int(round((
            previous["lines"][-1]["center_y"] +
            current["lines"][0]["center_y"]) / 2.0))
        px1, py1, px2, py2 = previous["bbox"]
        cx1, cy1, cx2, cy2 = current["bbox"]
        previous["bbox"] = (
            px1, py1, px2, max(py1 + 1, min(py2, boundary)))
        current["bbox"] = (
            cx1, min(cy2 - 1, max(cy1, boundary)), cx2, cy2)

    return paragraphs

def _region_label(region):
    """Return the stable user-facing title/heading/paragraph label."""
    return region.get("label", f"P{region.get('number', '?')}")


def _region_counts(regions):
    return {
        "titles": sum(
            region.get("region_type") == "page_title" for region in regions),
        "headings": sum(
            region.get("region_type") == "section_heading" for region in regions),
        "paragraphs": sum(
            region.get("region_type", "paragraph") == "paragraph"
            for region in regions),
    }


def paragraph_page_location(paragraph, image_shape):
    """Describe a paragraph's vertical location for audio guidance."""
    image_h = max(1, image_shape[0])
    _, y1, _, y2 = paragraph["bbox"]
    relative_y = ((y1 + y2) / 2.0) / image_h
    if relative_y < 0.20:
        return "at the top of the page"
    if relative_y < 0.40:
        return "in the upper part of the page"
    if relative_y < 0.62:
        return "in the middle of the page"
    if relative_y < 0.82:
        return "in the lower part of the page"
    return "at the bottom of the page"


def draw_paragraph_overlay(image, paragraphs, active_index=None):
    """Draw numbered paragraph boxes, highlighting the paragraph being read."""
    canvas = image.copy()
    for index, paragraph in enumerate(paragraphs):
        x1, y1, x2, y2 = paragraph["bbox"]
        active = index == active_index
        color = (0, 220, 255) if active else (255, 120, 0)
        thickness = 4 if active else 2
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        label = _region_label(paragraph)
        label_y = max(24, y1 - 8)
        cv2.putText(
            canvas, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, color, 2, cv2.LINE_AA)

    counts = _region_counts(paragraphs)
    if active_index is not None and 0 <= active_index < len(paragraphs):
        active_region = paragraphs[active_index]
        active_type = active_region.get("region_type", "paragraph")
        if active_type == "page_title":
            status = "READING TITLE"
        elif active_type == "section_heading":
            status = f"READING HEADING {active_region.get('heading_number', '')}".strip()
        else:
            status = (
                f"READING PARAGRAPH {active_region.get('paragraph_number', '?')} "
                f"OF {counts['paragraphs']}")
    else:
        status_parts = [f"{counts['paragraphs']} PARAGRAPHS"]
        if counts["titles"]:
            status_parts.append("TITLE")
        if counts["headings"]:
            status_parts.append(f"{counts['headings']} HEADINGS")
        status = " + ".join(status_parts) + " DETECTED"
    cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1], 620), 42), (20, 20, 20), -1)
    cv2.putText(
        canvas, status, (12, 29), cv2.FONT_HERSHEY_SIMPLEX,
        0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def show_paragraph_preview(image, paragraphs, active_index=None):
    """Display the OCR-coordinate image with paragraph mapping."""
    overlay = draw_paragraph_overlay(image, paragraphs, active_index)
    max_width, max_height = 1100, 800
    scale = min(
        1.0,
        max_width / max(1, overlay.shape[1]),
        max_height / max(1, overlay.shape[0]))
    if scale < 1.0:
        overlay = cv2.resize(
            overlay, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imshow("Captured Image (OCR Target)", overlay)
    cv2.waitKey(1)


# DEBUG — remove when done (saves overlay image + OCR text log per run for inspection)
def save_run_debug_outputs(
        run_count, img_path, ocr_image, paragraphs,
        extracted_text, language, spell_changes):
    """
    Save a paragraph overlay image and a structured text log for this run.
    Files are written to PARAGRAPH_TEST_DIR so nothing pollutes the main log.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_name = f"run_{run_count:03d}_{timestamp}"

    # ── 1. Paragraph overlay image ──────────────────────────────────────────
    if ocr_image is not None:
        overlay = draw_paragraph_overlay(ocr_image, paragraphs, active_index=None)
        overlay_path = os.path.join(
            PARAGRAPH_TEST_DIR, f"{base_name}_overlay.jpg")
        cv2.imwrite(overlay_path, overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  [DEBUG] Overlay saved  → {overlay_path}")

    # ── 2. Structured text log ───────────────────────────────────────────────
    log_path = os.path.join(PARAGRAPH_TEST_DIR, f"{base_name}_ocr.txt")
    try:
        with open(log_path, "w", encoding="utf-8") as fout:
            fout.write("=" * 70 + "\n")
            fout.write(f"RUN          : {run_count}\n")
            fout.write(f"TIMESTAMP    : {timestamp}\n")
            fout.write(f"IMAGE FILE   : {os.path.basename(img_path)}\n")
            fout.write(f"LANGUAGE     : {language}\n")
            counts = _region_counts(paragraphs)
            fout.write(f"TEXT REGIONS : {len(paragraphs)}\n")
            fout.write(f"PAGE TITLES  : {counts['titles']}\n")
            fout.write(f"HEADINGS     : {counts['headings']}\n")
            fout.write(f"PARAGRAPHS   : {counts['paragraphs']}\n")
            fout.write("=" * 70 + "\n\n")

            fout.write("── RAW EXTRACTED TEXT ──\n")
            fout.write(extracted_text + "\n\n")

            fout.write("── CLASSIFIED REGION DETAIL ──\n")
            for paragraph in paragraphs:
                fout.write("-" * 60 + "\n")
                fout.write(
                    f"{_region_label(paragraph)} | "
                    f"region_id={paragraph.get('region_id', paragraph['number'])} | "
                    f"type={paragraph.get('region_type', 'paragraph')} | "
                    f"lines={len(paragraph['lines'])} | "
                    f"conf={paragraph['confidence']:.3f} | "
                    f"font_scale={paragraph.get('font_scale', 1.0):.2f} | "
                    f"heading_evidence={paragraph.get('heading_evidence', 'unknown')} | "
                    f"break={paragraph.get('break_reason', 'unknown')} | "
                    f"normal_pitch={paragraph.get('typical_line_pitch', 0.0):.2f} | "
                    f"bbox={paragraph['bbox']}\n")
                fout.write(f"  SOURCE    : {paragraph['source_text']}\n")
                fout.write("  LAYOUT LINES:\n")
                for line_number, line in enumerate(
                        paragraph["lines"], start=1):
                    fout.write(
                        f"    L{line_number:02d} | bbox={line['bbox']} | "
                        f"pitch={line.get('paragraph_pitch', 0.0):.2f} | "
                        f"gap_ratio={line.get('paragraph_gap_ratio', 0.0):.2f} | "
                        f"indent={line.get('paragraph_indent_delta', 0.0):.2f} | "
                        f"font_height={line.get('font_height', line['height']):.2f} | "
                        f"font_scale={line.get('font_scale', 1.0):.2f} | "
                        f"heading={line.get('heading_evidence', 'unknown')} | "
                        f"decision={line.get('paragraph_break_reason', 'unknown')} | "
                        f"text={line['text']}\n")
                if language != "english" and "translated_text" in paragraph:
                    fout.write(
                        f"  TRANSLATED: {paragraph['translated_text']}\n")
                if "spoken_text" in paragraph:
                    fout.write(
                        f"  SPOKEN    : {paragraph['spoken_text']}\n")
                if paragraph.get("spell_changes"):
                    fout.write(
                        f"  SPELL FIXES: {paragraph['spell_changes']}\n")

            if spell_changes:
                fout.write("\n── SPELL CORRECTIONS SUMMARY ──\n")
                for region_label, original, replacement in spell_changes:
                    fout.write(
                        f"  {region_label}: {original!r} → {replacement!r}\n")

            fout.write("\n" + "=" * 70 + "\n")
        print(f"  [DEBUG] OCR log saved  → {log_path}")
    except Exception as save_err:
        print(f"  [DEBUG] Could not save OCR log: {save_err}")


def run_ocr(ocr_engine, img_path, book_mode, unwarper):
    t0 = time.time()
    if book_mode and unwarper is not None:
        print("[STAGE 1] Unwarping curved page …")
        unwarp_result = unwarper.predict(img_path, batch_size=1)
        for res in unwarp_result:
            unwarped_img = res["doctr_img"]
        temp_path = os.path.join(PIPELINE_DIR, "_temp_unwarped.jpg")
        cv2.imwrite(temp_path, unwarped_img)
        img_path = temp_path

    print("[STAGE 1] Preprocessing image …")
    processed = preprocess_image(img_path)
    if processed is None:
        return "", [], None, time.time() - t0
    print("[STAGE 1] Running PaddleOCR …")
    ocr_input = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    with OCR_LOCK:
        result = ocr_engine.predict(ocr_input)
    raw_lines = _extract_ocr_lines(result, ocr_input.shape)
    lines = merge_ocr_fragments_into_lines(raw_lines)
    paragraphs = group_ocr_lines_into_paragraphs(lines, ocr_input.shape)
    extracted = " ".join(
        paragraph["source_text"] for paragraph in paragraphs)

    for f in [os.path.join(PIPELINE_DIR, "_temp_unwarped.jpg")]:
        if os.path.exists(f):
            os.remove(f)

    elapsed = time.time() - t0
    counts = _region_counts(paragraphs)
    print(
        f"[STAGE 1] OCR complete in {elapsed:.3f}s  "
        f"({len(raw_lines)} detections -> {len(lines)} lines, "
        f"{len(paragraphs)} regions: {counts['titles']} title, "
        f"{counts['headings']} headings, {counts['paragraphs']} paragraphs)")
    return extracted, paragraphs, ocr_input, elapsed


def run_translation_nllb(text, language, translator, tokenizer):
    if translator is None or tokenizer is None:
        print("[STAGE 2] Skipping translation (English source)")
        return text, 0.0
    print("[STAGE 2] Translating → English (NLLB 1.3B) …")
    t0 = time.time()
    src_lang_code      = PIPELINE_LANG_TO_NLLB[language]
    src_token          = NLLB_LANG_MAP[src_lang_code]
    tgt_token          = NLLB_LANG_MAP["en"]
    tokenizer.src_lang = src_token

    if src_lang_code == "zh":
        raw_chunks = re.split(r'(。|？|！)', text)
        chunks = []
        for i in range(0, len(raw_chunks)-1, 2):
            chunks.append(raw_chunks[i] + raw_chunks[i+1])
        if len(raw_chunks) % 2 != 0 and raw_chunks[-1].strip():
            chunks.append(raw_chunks[-1])
    else:
        chunks = re.split(r'(?<=[.!?])\s+', text)

    chunks = [c.strip() for c in chunks if c.strip()]
    if not chunks:
        return text, time.time() - t0

    tokenized_batch = []
    for chunk in chunks:
        tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(chunk))
        tokenized_batch.append(tokens)
    if not tokenized_batch:
        return text, time.time() - t0

    target_prefixes = [[tgt_token]] * len(tokenized_batch)
    results = translator.translate_batch(tokenized_batch, target_prefix=target_prefixes)

    translated_sentences = []
    for res in results:
        output_tokens  = res.hypotheses[0]
        raw_decoded    = tokenizer.decode(tokenizer.convert_tokens_to_ids(output_tokens))
        clean_sentence = raw_decoded.replace(tgt_token, "").strip()
        translated_sentences.append(clean_sentence)

    result  = " ".join(translated_sentences)
    elapsed = time.time() - t0
    print(f"[STAGE 2] Translation complete in {elapsed:.3f}s")
    return result, elapsed


def _split_translation_units(text, src_lang_code):
    """Split a paragraph at sentence boundaries without losing punctuation."""
    if src_lang_code == "zh":
        units = re.split(r"(?<=[。？！!?])", text)
    else:
        units = re.split(r"(?<=[.!?])\s+", text)
    return [unit.strip() for unit in units if unit.strip()]


def _fit_translation_unit(unit, tokenizer, src_lang_code, max_tokens=400):
    """Keep every NLLB request safely below its context limit."""
    if len(tokenizer.encode(unit)) <= max_tokens:
        return [unit]

    is_chinese = src_lang_code == "zh"
    pieces = list(unit) if is_chinese else unit.split()
    joiner = "" if is_chinese else " "
    chunks = []
    current = []
    for piece in pieces:
        candidate = joiner.join(current + [piece])
        if current and len(tokenizer.encode(candidate)) > max_tokens:
            chunks.append(joiner.join(current).strip())
            current = [piece]
        else:
            current.append(piece)
    if current:
        chunks.append(joiner.join(current).strip())
    return [chunk for chunk in chunks if chunk]


def run_translation_nllb_paragraphs(
        paragraphs, language, translator, tokenizer):
    """
    Translate every paragraph in one efficient NLLB batch while preserving IDs.

    Paragraphs may be split into sentence/token-safe chunks internally, but all
    translated chunks are reassembled into their original paragraph before TTS.
    """
    translated_paragraphs = [dict(paragraph) for paragraph in paragraphs]
    if translator is None or tokenizer is None:
        print("[STAGE 2] Skipping translation (English source)")
        for paragraph in translated_paragraphs:
            paragraph["translated_text"] = paragraph["source_text"]
        return translated_paragraphs, 0.0

    print(
        f"[STAGE 2] Translating {len(paragraphs)} classified region(s) -> English "
        f"in one NLLB batch …")
    t0 = time.time()
    src_lang_code = PIPELINE_LANG_TO_NLLB[language]
    src_token = NLLB_LANG_MAP[src_lang_code]
    tgt_token = NLLB_LANG_MAP["en"]
    tokenizer.src_lang = src_token

    flat_chunks = []
    chunk_owners = []
    for paragraph_index, paragraph in enumerate(translated_paragraphs):
        sentence_units = _split_translation_units(
            paragraph["source_text"], src_lang_code)
        if not sentence_units:
            sentence_units = [paragraph["source_text"]]
        for unit in sentence_units:
            for chunk in _fit_translation_unit(
                    unit, tokenizer, src_lang_code):
                flat_chunks.append(chunk)
                chunk_owners.append(paragraph_index)

    if not flat_chunks:
        for paragraph in translated_paragraphs:
            paragraph["translated_text"] = paragraph["source_text"]
        return translated_paragraphs, time.time() - t0

    tokenized_batch = [
        tokenizer.convert_ids_to_tokens(tokenizer.encode(chunk))
        for chunk in flat_chunks
    ]
    target_prefixes = [[tgt_token]] * len(tokenized_batch)
    results = translator.translate_batch(
        tokenized_batch, target_prefix=target_prefixes)

    translated_by_paragraph = [
        [] for _ in translated_paragraphs]
    for owner, result in zip(chunk_owners, results):
        output_tokens = result.hypotheses[0]
        raw_decoded = tokenizer.decode(
            tokenizer.convert_tokens_to_ids(output_tokens))
        clean_text = raw_decoded.replace(tgt_token, "").strip()
        if clean_text:
            translated_by_paragraph[owner].append(clean_text)

    for index, paragraph in enumerate(translated_paragraphs):
        translated_text = " ".join(translated_by_paragraph[index]).strip()
        paragraph["translated_text"] = (
            translated_text if translated_text else paragraph["source_text"])

    elapsed = time.time() - t0
    print(
        f"[STAGE 2] Region translation complete in {elapsed:.3f}s "
        f"({len(flat_chunks)} translation chunks)")
    return translated_paragraphs, elapsed


# ═══════════════════════════════════════════════════════════════════════════════
#  CONSERVATIVE OCR SPELL CORRECTION — SymSpell candidate generator
# ═══════════════════════════════════════════════════════════════════════════════

# These terms are always preserved. Generic rules below also preserve every
# acronym, mixed-case identifier, number, hyphenated term, and bracketed span.
_SPELL_PROTECTED_TERMS = {
    "airworthiness", "aerodynamic", "avionics", "barometer", "barometers",
    "bidirectional", "brushless", "chipset", "chipsets", "dynamometer",
    "electromagnetic", "fuselage", "maneuvering", "manoeuvring", "multirotor",
    "odometry", "powertrain", "telemetry", "topologically", "transmitter",
    "transmitters", "unwarping", "voxelization",
}

_BRACKETED_SPAN_RE = re.compile(
    r"(\[[^\]]*\]|\([^)]*\)|\{[^}]*\}|<[^>]*>)"
)
_PLAIN_TOKEN_RE = re.compile(r"^([^A-Za-z]*)([A-Za-z]+)([^A-Za-z]*)$")

# A correction must be a frequent dictionary word and a visually plausible
# OCR edit. These thresholds intentionally prefer a missed correction over a
# false replacement that changes the document's meaning.
_SPELL_MIN_WORD_LENGTH = 6
_SPELL_MIN_CANDIDATE_COUNT = 300_000
_SPELL_MIN_DOMINANCE_RATIO = 100


def load_spell_corrector():
    """Load SymSpell as a candidate finder; compound rewriting is disabled."""
    t0 = time.time()
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dictionary_path = os.path.join(
        os.path.dirname(symspellpy.__file__),
        "frequency_dictionary_en_82_765.txt")
    if not sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1):
        raise RuntimeError(f"Could not load SymSpell dictionary: {dictionary_path}")
    print(f"[LOAD] Conservative SymSpell dictionary loaded in {time.time()-t0:.2f}s")
    return sym_spell


def _is_abbreviation_or_identifier(word):
    """Return True for acronyms, proper names, and mixed-case identifiers."""
    if len(word) <= 4:
        return True
    if word[0].isupper():
        return True
    if any(char.isupper() for char in word[1:]):
        return True
    if re.fullmatch(r"[A-Z]{2,}[a-z]?", word):
        return True
    return False


def _is_plausible_ocr_edit(source, candidate):
    """Allow only a small set of common visual OCR confusions."""
    source = source.lower()
    candidate = candidate.lower()
    if source == candidate:
        return False

    # Multi-character glyph confusions such as "intemal" -> "internal".
    def glyph_key(value):
        return value.replace("rn", "m").replace("cl", "d").replace("vv", "w")

    if glyph_key(source) == glyph_key(candidate):
        return True

    # One visually similar character substitution.
    if len(source) == len(candidate):
        differences = [
            (left, right)
            for left, right in zip(source, candidate)
            if left != right
        ]
        if len(differences) == 1:
            allowed_groups = (
                frozenset(("i", "l")),
                frozenset(("i", "j")),
                frozenset(("c", "e")),
                frozenset(("u", "v")),
                frozenset(("n", "r")),
                frozenset(("t", "f")),
            )
            return frozenset(differences[0]) in allowed_groups

    return False


def _correct_plain_segment(sym_spell, segment, changes):
    """Correct unprotected whitespace-delimited tokens without reformatting."""
    pieces = re.split(r"(\s+)", segment)
    for index, token in enumerate(pieces):
        if not token or token.isspace():
            continue

        # Never touch partial/malformed brackets, codes, measurements,
        # hyphenated technical terms (Li-Ion/Li-on), or slash-separated terms.
        if any(char in token for char in "[](){}<>0123456789-/\\+_=|@#%"):
            continue

        match = _PLAIN_TOKEN_RE.fullmatch(token)
        if match is None:
            continue
        prefix, word, suffix = match.groups()
        lower_word = word.lower()

        if len(word) < _SPELL_MIN_WORD_LENGTH:
            continue
        if _is_abbreviation_or_identifier(word):
            continue
        if lower_word in _SPELL_PROTECTED_TERMS:
            continue

        # A valid dictionary word is never rewritten. This prevents contextual
        # false positives such as "radio" -> "audio" or "array" -> "army".
        if lower_word in sym_spell.words:
            continue

        suggestions = sym_spell.lookup(
            lower_word,
            Verbosity.ALL,
            max_edit_distance=2,
            include_unknown=False)
        eligible = [
            suggestion for suggestion in suggestions
            if suggestion.distance in (1, 2)
            and suggestion.term.isalpha()
            and suggestion.count >= _SPELL_MIN_CANDIDATE_COUNT
            and _is_plausible_ocr_edit(lower_word, suggestion.term)
        ]
        if not eligible:
            continue

        eligible.sort(key=lambda suggestion: suggestion.count, reverse=True)
        best = eligible[0]
        if (len(eligible) > 1 and
                best.count < eligible[1].count * _SPELL_MIN_DOMINANCE_RATIO):
            continue

        replacement = best.term
        pieces[index] = f"{prefix}{replacement}{suffix}"
        changes.append((word, replacement))

    return "".join(pieces)


def run_spell_correction(sym_spell, text):
    """
    Apply only high-confidence OCR corrections.

    Text inside brackets is copied byte-for-byte. Abbreviations, identifiers,
    numbers, technical codes, and hyphenated words are never passed to
    SymSpell. Returns both the corrected text and an auditable change list.
    """
    if sym_spell is None or not text or not text.strip():
        return text, []

    changes = []
    chunks = _BRACKETED_SPAN_RE.split(text)
    corrected_chunks = []
    for chunk in chunks:
        if not chunk:
            continue
        if _BRACKETED_SPAN_RE.fullmatch(chunk):
            corrected_chunks.append(chunk)
        else:
            corrected_chunks.append(
                _correct_plain_segment(sym_spell, chunk, changes))
    return "".join(corrected_chunks), changes


def run_tts(tts_module, text, key_queue=None, event_queue=None):
    """Speak text. Press any key on PC or Pi to stop early."""
    print("[STAGE 3] Speaking … (press any key to stop)")
    t0 = time.time()

    # Clear out any old keystrokes/commands before starting playback
    if key_queue is not None:
        try:
            while not key_queue.empty():
                key_queue.get_nowait()
        except queue.Empty:
            pass

    if event_queue is not None:
        try:
            while not event_queue.empty():
                event_queue.get_nowait()
        except queue.Empty:
            pass

    tts_module.speak(text)

    stopped = False
    while tts_module.is_speaking():
        # Check PC keyboard queue
        if key_queue is not None and not key_queue.empty():
            try:
                while not key_queue.empty():
                    key_queue.get_nowait()
                tts_module.stop()
                stopped = True
                break
            except queue.Empty:
                pass

        # Check Pi command event queue
        if event_queue is not None and not event_queue.empty():
            try:
                stop_signal = False
                while not event_queue.empty():
                    evt = event_queue.get_nowait()
                    if evt.get("cmd") == "capture_from_pi" or evt.get("event") == "disconnect":
                        stop_signal = True
                if stop_signal:
                    tts_module.stop()
                    stopped = True
                    break
            except queue.Empty:
                pass

        time.sleep(0.05)

    tts_module.wait_until_done()

    # Clear any keystrokes/commands typed during playback to prevent queue build-up looping
    if key_queue is not None:
        try:
            while not key_queue.empty():
                key_queue.get_nowait()
        except queue.Empty:
            pass

    if event_queue is not None:
        try:
            while not event_queue.empty():
                event_queue.get_nowait()
        except queue.Empty:
            pass

    elapsed = time.time() - t0
    if stopped:
        print(f"[STAGE 3] TTS stopped after {elapsed:.3f}s")
    else:
        print(f"[STAGE 3] TTS finished in {elapsed:.3f}s")
    return elapsed


def _clear_queue_safely(target_queue):
    if target_queue is None:
        return
    try:
        while not target_queue.empty():
            target_queue.get_nowait()
    except queue.Empty:
        pass


def _speak_segment_with_stop(
        tts_module, text, key_queue=None, event_queue=None):
    """Speak one region; A toggles pause/resume and other keys still stop."""
    tts_module.speak(text)
    stopped = False
    last_pause_key_event = 0.0
    while tts_module.is_speaking():
        if key_queue is not None:
            try:
                while True:
                    key = key_queue.get_nowait()
                    if key == "a":
                        now = time.monotonic()
                        # Update the timestamp even for ignored repeats, so a
                        # held A key cannot toggle repeatedly.
                        repeated_key = now - last_pause_key_event < 0.55
                        last_pause_key_event = now
                        if repeated_key:
                            continue
                        if tts_module.is_paused():
                            tts_module.resume()
                        else:
                            tts_module.pause()
                    else:
                        # Preserve box3d2 behaviour: every non-A key stops.
                        # In particular, the first S stops TTS and a fresh
                        # second S is still handled by main as a new capture.
                        _clear_queue_safely(key_queue)
                        tts_module.stop()
                        stopped = True
                        break
                if stopped:
                    break
            except queue.Empty:
                pass

        if event_queue is not None and not event_queue.empty():
            stop_signal = False
            try:
                while not event_queue.empty():
                    event = event_queue.get_nowait()
                    if (event.get("cmd") == "capture_from_pi" or
                            event.get("event") == "disconnect"):
                        stop_signal = True
            except queue.Empty:
                pass
            if stop_signal:
                tts_module.stop()
                stopped = True
                break
        time.sleep(0.05)

    tts_module.wait_until_done()
    return stopped


def run_tts_paragraphs(
        tts_module, paragraphs, ocr_image,
        key_queue=None, event_queue=None):
    """Announce, highlight, and speak classified regions in reading order."""
    readable = [
        paragraph for paragraph in paragraphs
        if paragraph.get("spoken_text", "").strip()
    ]
    if not readable:
        return 0.0

    counts = _region_counts(readable)
    print(
        f"[STAGE 3] Speaking {len(readable)} classified region(s) in reading order "
        f"… (A = pause/resume, S = stop)")
    t0 = time.time()
    _clear_queue_safely(key_queue)
    _clear_queue_safely(event_queue)
    stopped = False

    for index, paragraph in enumerate(readable):
        show_paragraph_preview(ocr_image, readable, active_index=index)
        location = paragraph_page_location(paragraph, ocr_image.shape)
        region_type = paragraph.get("region_type", "paragraph")
        label = _region_label(paragraph)
        if region_type == "page_title":
            announcement = f"Title, {location}."
        elif region_type == "section_heading":
            announcement = (
                f"Heading {paragraph.get('heading_number', '')}, "
                f"{location}.")
        else:
            announcement = (
                f"Paragraph {paragraph.get('paragraph_number', '?')} "
                f"of {counts['paragraphs']}, {location}.")
        spoken_segment = f"{announcement} {paragraph['spoken_text'].strip()}"
        print(f"[STAGE 3] {label} — {location}")
        if _speak_segment_with_stop(
                tts_module, spoken_segment, key_queue, event_queue):
            stopped = True
            break

    show_paragraph_preview(ocr_image, readable, active_index=None)
    _clear_queue_safely(key_queue)
    _clear_queue_safely(event_queue)
    elapsed = time.time() - t0
    if stopped:
        print(f"[STAGE 3] Region TTS stopped after {elapsed:.3f}s")
    else:
        print(f"[STAGE 3] Region TTS finished in {elapsed:.3f}s")
    return elapsed


# ══════════════════════════════════════════════════════════════════════════════
#  RECEIVER THREAD  — keeps the latest frame + capture signals in queues
# ══════════════════════════════════════════════════════════════════════════════
class FrameHolder:
    """Thread-safe holder for the most recent camera frame from the Pi."""
    def __init__(self):
        self._frame = None
        self._lock  = threading.Lock()
        self._sequence = 0
        self._timestamp = 0.0

    def update(self, frame):
        with self._lock:
            self._frame = frame
            self._sequence += 1
            self._timestamp = time.monotonic()
            return self._sequence, self._timestamp

    def get(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def get_snapshot(self):
        """Return one frame together with the ID/time belonging to that frame."""
        with self._lock:
            if self._frame is None:
                return None, self._sequence, self._timestamp
            # receiver_loop replaces this array but never mutates it. Returning
            # the stable reference avoids copying a 1080p frame on every preview
            # iteration; the analysis worker also treats it as immutable.
            return self._frame, self._sequence, self._timestamp


def receiver_loop(sock, frame_holder, event_queue, debug_recorder=None):
    """Background thread: reads messages from Pi, updates frame_holder and event_queue."""
    while True:
        try:
            msg_type, data = recv_msg(sock)
            if msg_type == MSG_FRAME:
                sequence, frame_timestamp = frame_holder.update(data)
                _safe_debug_call(
                    debug_recorder, "submit_live_frame",
                    data, sequence, frame_timestamp)
            elif msg_type == MSG_JSON:
                event_queue.put(data)
                _safe_debug_call(
                    debug_recorder, "record_event",
                    "pi_message", message=data)
        except (ConnectionError, OSError, ValueError):
            event_queue.put({"event": "disconnect"})
            _safe_debug_call(
                debug_recorder, "record_event", "pi_disconnected")
            break


# ══════════════════════════════════════════════════════════════════════════════
#  KEYBOARD LISTENER  (PC side — A pauses TTS, S captures/stops, Q quits)
# ══════════════════════════════════════════════════════════════════════════════
def start_keyboard_listener(key_queue):
    """Start a background thread that reads keyboard input on PC."""
    def _listener_windows():
        import msvcrt
        while True:
            if msvcrt.kbhit():
                key = msvcrt.getch().decode("utf-8", errors="ignore").lower()
                key_queue.put(key)
            time.sleep(0.02)

    def _listener_unix():
        import termios, tty, select
        old = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())
            while True:
                if select.select([sys.stdin], [], [], 0.02)[0]:
                    key = sys.stdin.read(1).lower()
                    key_queue.put(key)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)

    t = threading.Thread(
        target=_listener_windows if sys.platform == "win32" else _listener_unix,
        daemon=True)
    t.start()


# ══════════════════════════════════════════════════════════════════════════════
#  TCP SERVER
# ══════════════════════════════════════════════════════════════════════════════
def start_server(port):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', port))
    server.listen(1)
    print(f"\n[SERVER] Listening on port {port} …")
    print("  Waiting for Pi to connect …\n")
    client, addr = server.accept()
    print(f"[SERVER] ✅ Pi connected from {addr}")
    client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    return server, client


# ══════════════════════════════════════════════════════════════════════════════
#  FRAME QUALITY SCORING
# ══════════════════════════════════════════════════════════════════════════════
QUALITY_THRESHOLD = 50  # retained for the legacy/post-save display only

# Box4b capture analysis is resolution-normalized and calibrated on the supplied
# 1920x1080 captures plus the recorded head-mounted sequence. These thresholds
# apply only to the 960-pixel analysis copy; the OCR target remains untouched.
CAPTURE_ANALYSIS_WIDTH = 960
CAPTURE_ANALYSIS_INTERVAL = 0.90
CAPTURE_REQUIRED_DISTINCT = 3
CAPTURE_STABILITY_WINDOW = 4
CAPTURE_MIN_STABLE_SECONDS = 1.25
CAPTURE_MAX_ANALYSIS_AGE = 1.35
CAPTURE_COVERAGE_HISTORY = 12
CAPTURE_COVERAGE_RATIO = 0.78
CAPTURE_FOCUS_BAND_MIN = 110.0
CAPTURE_LINE_FOCUS_MIN = 120.0
CAPTURE_LIGHT_MIN = 105.0
CAPTURE_LIGHT_MAX = 247.0
CAPTURE_LIGHT_TILE_STD_MAX = 24.0
CAPTURE_PAPER_BORDER_FRACTION = 0.20
CAPTURE_PAPER_BORDER_RUN = 0.20
# A text box needs only a small fraction of one detected line-height between it
# and the image edge. This scales correctly for 16/18 pt text, camera distance,
# and perspective. It also rejects genuinely clipped text while ignoring blank
# A4 paper margins and the physical paper contour.
CAPTURE_TEXT_CLEARANCE_LINES = 0.50
CAPTURE_MIN_LINE_HEIGHT_RATIO = 0.013
CAPTURE_EXTRA_TEXT_FOCUS_MIN = 80.0
CAPTURE_OUTSIDE_TEXT_MEAN_MIN = 3.60
CAPTURE_OUTSIDE_TEXT_FRACTION_MIN = 0.080
# Inter-sample motion is measured about 0.90 seconds apart. Normal head-mounted
# parallax can move a still-sharp page several percent between those samples;
# actual blur remains guarded independently by the page-band and line focus
# gates below.
CAPTURE_MOTION_SHIFT_MAX = 0.065
CAPTURE_MOTION_GEOMETRY_MAX = 0.075
CAPTURE_MOTION_AREA_CHANGE_MAX = 0.35
CAPTURE_MOTION_ANGLE_MAX = 8.0


def score_frame_quality(frame):
    """Compatibility score used after save; capture decisions use hard gates."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness_pts = int(np.clip((laplacian_var - 50.0) / 250.0 * 40.0, 0, 40))
    mean_brightness = float(np.mean(gray))
    if 80 <= mean_brightness <= 180:
        brightness_pts = 30
    elif mean_brightness < 80:
        brightness_pts = max(0, int(mean_brightness / 80 * 30))
    else:
        brightness_pts = max(0, int((255 - mean_brightness) / 75 * 30))
    mid_h, mid_w = h // 2, w // 2
    quadrants = [gray[:mid_h, :mid_w], gray[:mid_h, mid_w:],
                 gray[mid_h:, :mid_w], gray[mid_h:, mid_w:]]
    quad_std = float(np.std([float(np.mean(q)) for q in quadrants]))
    evenness_pts = int(np.clip((40.0 - quad_std) / 35.0 * 30.0, 0, 30))
    score = min(100, sharpness_pts + brightness_pts + evenness_pts)
    details = {
        "sharpness": round(laplacian_var, 1),
        "sharpness_pts": sharpness_pts,
        "brightness": round(mean_brightness, 1),
        "brightness_pts": brightness_pts,
        "evenness_std": round(quad_std, 1),
        "evenness_pts": evenness_pts,
    }
    return score, details


# ══════════════════════════════════════════════════════════════════════════════
#  TEXT REGION DETECTION & OUTER PAGE BOX GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
def _longest_true_run(values):
    longest = current = 0
    for value in values:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _extract_detection_boxes(ocr_result, work):
    """Convert Paddle polygons into geometry/local-background records."""
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    boxes = []
    for result_item in ocr_result:
        try:
            polygons = result_item.get("dt_polys", [])
        except AttributeError:
            polygons = result_item["dt_polys"] if "dt_polys" in result_item else []
        for polygon in polygons:
            poly = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            if len(poly) < 4:
                continue
            (cx, cy), (rect_w, rect_h), angle = cv2.minAreaRect(poly)
            if rect_w < rect_h:
                rect_w, rect_h = rect_h, rect_w
                angle += 90.0
            angle = ((angle + 90.0) % 180.0) - 90.0
            x1, y1 = poly.min(axis=0)
            x2, y2 = poly.max(axis=0)
            xa, xb = max(0, int(x1) - 4), min(w, int(x2) + 5)
            ya, yb = max(0, int(y1) - 4), min(h, int(y2) + 5)
            local = gray[ya:yb, xa:xb]
            background = float(np.percentile(local, 80)) if local.size else 0.0
            boxes.append({
                "poly": poly,
                "cx": float(cx), "cy": float(cy),
                "w": float(rect_w), "h": float(rect_h),
                "angle": float(angle),
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
                "bg80": background,
            })
    return boxes


def _group_document_rows(boxes):
    if not boxes:
        return [], 0.0, 0.0
    median_height = float(np.median([box["h"] for box in boxes]))
    rows = []
    for box in sorted(boxes, key=lambda item: item["cy"]):
        if not rows or box["cy"] - rows[-1][-1]["cy"] > 0.65 * median_height:
            rows.append([box])
        else:
            rows[-1].append(box)
    centres = np.asarray([np.mean([item["cy"] for item in row]) for row in rows])
    gaps = np.diff(centres)
    median_gap = float(np.median(gaps)) if len(gaps) else 0.0
    max_gap_ratio = float(np.max(gaps) / max(median_gap, 1e-6)) if len(gaps) else 0.0
    return rows, median_gap, max_gap_ratio


def _dominant_document_cluster(boxes, width, height):
    """Keep the dense paper text and discard monitor/keyboard OCR outliers."""
    candidates = [
        box for box in boxes
        if (box["bg80"] >= 95.0 and 4.0 <= box["h"] <= 0.09 * height
            and box["w"] >= 1.15 * box["h"])
    ]
    if not candidates:
        return None

    angle_histogram = np.zeros(36, dtype=np.float64)
    for box in candidates:
        angle = max(-44.9, min(44.9, box["angle"]))
        angle_histogram[int((angle + 45.0) // 2.5)] += min(box["w"], 0.35 * width)
    dominant_angle = -45.0 + (int(np.argmax(angle_histogram)) + 0.5) * 2.5
    angle_filtered = [
        box for box in candidates
        if abs(((box["angle"] - dominant_angle + 90.0) % 180.0) - 90.0) <= 13.0
    ]
    if len(angle_filtered) >= 3:
        candidates = angle_filtered

    median_height = float(np.median([box["h"] for box in candidates]))
    parent = list(range(len(candidates)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first, second):
        first, second = find(first), find(second)
        if first != second:
            parent[second] = first

    for first_index, first in enumerate(candidates):
        ax1, _, ax2, _ = first["bbox"]
        for second_index in range(first_index + 1, len(candidates)):
            second = candidates[second_index]
            bx1, _, bx2, _ = second["bbox"]
            delta_y = abs(first["cy"] - second["cy"])
            delta_x = abs(first["cx"] - second["cx"])
            overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1))
            minimum_width = max(1.0, min(first["w"], second["w"]))
            horizontally_close = delta_x <= max(
                0.19 * width, 0.65 * (first["w"] + second["w"]))
            if (delta_y <= max(7.0 * median_height, 0.09 * height)
                    and (overlap / minimum_width >= 0.08 or horizontally_close)):
                union(first_index, second_index)

    groups = {}
    for index, box in enumerate(candidates):
        groups.setdefault(find(index), []).append(box)

    def cluster_score(group):
        x_values = [value for box in group for value in (box["bbox"][0], box["bbox"][2])]
        y_values = [value for box in group for value in (box["bbox"][1], box["bbox"][3])]
        span = ((max(x_values) - min(x_values)) * (max(y_values) - min(y_values))
                / float(width * height))
        return (2.0 * len(group)
                + sum(min(box["w"] / width, 0.30) for box in group)
                + 10.0 * min(span, 0.50)
                + float(np.median([box["bg80"] for box in group])) / 80.0)

    group = max(groups.values(), key=cluster_score)
    x1 = min(box["bbox"][0] for box in group)
    y1 = min(box["bbox"][1] for box in group)
    x2 = max(box["bbox"][2] for box in group)
    y2 = max(box["bbox"][3] for box in group)
    rows, median_gap, max_gap_ratio = _group_document_rows(group)
    return {
        "boxes": group,
        "rect": (float(x1), float(y1), float(x2), float(y2)),
        "rows": rows,
        "row_count": len(rows),
        "median_height": float(np.median([box["h"] for box in group])),
        "median_gap": median_gap,
        "max_gap_ratio": max_gap_ratio,
        "angle": float(np.median([box["angle"] for box in group])),
        "background": float(np.median([box["bg80"] for box in group])),
    }


def _strict_paper_component(work, text_rect):
    """Return strict low-chroma paper evidence seeded by the document text."""
    h, w = work.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in text_rect]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 - x1 < 20 or y2 - y1 < 20:
        return None

    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
    inner_hsv, inner_lab = hsv[y1:y2, x1:x2], lab[y1:y2, x1:x2]
    saturation, value = inner_hsv[:, :, 1], inner_hsv[:, :, 2]
    sample_mask = ((saturation <= np.percentile(saturation, 55))
                   & (value >= np.percentile(value, 62)))
    samples, sample_hsv = inner_lab[sample_mask], inner_hsv[sample_mask]
    if len(samples) < 200:
        samples = inner_lab.reshape(-1, 3)
        sample_hsv = inner_hsv.reshape(-1, 3)
    median_lab = np.median(samples, axis=0).astype(np.float32)
    median_s = float(np.median(sample_hsv[:, 1]))
    median_v = float(np.median(sample_hsv[:, 2]))

    lightness = lab[:, :, 0].astype(np.float32)
    channel_a = lab[:, :, 1].astype(np.float32)
    channel_b = lab[:, :, 2].astype(np.float32)
    saturation_all = hsv[:, :, 1].astype(np.float32)
    value_all = hsv[:, :, 2].astype(np.float32)
    chroma_distance = np.hypot(
        channel_a - median_lab[1], channel_b - median_lab[2])
    saturation_limit = min(35.0, max(20.0, median_s + 15.0))
    mask = (
        (saturation_all <= saturation_limit)
        & (chroma_distance <= 13.0)
        & (value_all >= max(70.0, median_v - 100.0))
        & (lightness >= max(65.0, median_lab[0] - 105.0))
    ).astype(np.uint8) * 255

    kernel_size = max(7, int(round(w / 105.0)) | 1)
    closed = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        np.ones((kernel_size, kernel_size), np.uint8), iterations=2)
    closed = cv2.morphologyEx(
        closed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    seed_x1 = x1 + int(0.08 * (x2 - x1))
    seed_x2 = x2 - int(0.08 * (x2 - x1))
    seed_y1 = y1 + int(0.03 * (y2 - y1))
    seed_y2 = y2 - int(0.03 * (y2 - y1))
    best_label, best_score = None, -1e30
    seed_area = max(1, (seed_x2 - seed_x1) * (seed_y2 - seed_y1))
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 0.018 * w * h:
            continue
        overlap = int(np.count_nonzero(
            labels[seed_y1:seed_y2, seed_x1:seed_x2] == label))
        if overlap < 100:
            continue
        score = 7.0 * overlap / seed_area + 0.4 * area / float(w * h)
        if score > best_score:
            best_label, best_score = label, score
    if best_label is None:
        return None

    selected = (labels == best_label).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        selected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    bbox = cv2.boundingRect(hull)
    strip = max(4, int(round(0.012 * min(w, h))))
    fractions = {
        "L": float(np.count_nonzero(selected[:, :strip])) / float(h * strip),
        "T": float(np.count_nonzero(selected[:strip, :])) / float(w * strip),
        "R": float(np.count_nonzero(selected[:, w - strip:])) / float(h * strip),
        "B": float(np.count_nonzero(selected[h - strip:, :])) / float(w * strip),
    }
    active_profiles = {
        "L": np.mean(selected[:, :strip] > 0, axis=1) > 0.40,
        "T": np.mean(selected[:strip, :] > 0, axis=0) > 0.40,
        "R": np.mean(selected[:, w - strip:] > 0, axis=1) > 0.40,
        "B": np.mean(selected[h - strip:, :] > 0, axis=0) > 0.40,
    }
    runs = {
        "L": _longest_true_run(active_profiles["L"]) / float(h),
        "T": _longest_true_run(active_profiles["T"]) / float(w),
        "R": _longest_true_run(active_profiles["R"]) / float(h),
        "B": _longest_true_run(active_profiles["B"]) / float(w),
    }
    return {
        "contour": contour,
        "hull": hull,
        "bbox": tuple(int(value) for value in bbox),
        "fractions": fractions,
        "runs": runs,
        "paper_brightness": median_v,
    }


def _document_quality(work, cluster):
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    x1, y1, x2, y2 = cluster["rect"]
    pad_x = max(3, int(0.06 * (x2 - x1)))
    pad_y = max(3, int(0.04 * (y2 - y1)))
    xa, xb = max(0, int(x1) - pad_x), min(w, int(x2) + pad_x)
    ya, yb = max(0, int(y1) - pad_y), min(h, int(y2) + pad_y)
    roi = gray[ya:yb, xa:xb]
    if roi.size == 0:
        return {
            "focus_ok": False, "lighting_ok": False,
            "too_dark": False, "glare": False,
            "band_lap_min": 0.0, "band_laps": [0.0, 0.0, 0.0],
            "line_lap_p20": 0.0, "light_median": 0.0,
            "light_tile_std": 999.0, "roi_rect": (xa, ya, xb, yb),
        }

    bands = [band for band in np.array_split(roi, 3, axis=0) if band.size]
    band_laps = [float(cv2.Laplacian(band, cv2.CV_64F).var()) for band in bands]
    line_laps = []
    for box in cluster["boxes"]:
        bx1, by1, bx2, by2 = box["bbox"]
        lx1, lx2 = max(0, int(bx1) - 2), min(w, int(bx2) + 3)
        ly1, ly2 = max(0, int(by1) - 2), min(h, int(by2) + 3)
        line_roi = gray[ly1:ly2, lx1:lx2]
        if line_roi.size:
            line_laps.append(float(cv2.Laplacian(
                line_roi, cv2.CV_64F).var()))
    line_lap_p20 = float(np.percentile(line_laps, 20)) if line_laps else 0.0

    tile_backgrounds = []
    for row_band in np.array_split(roi, 3, axis=0):
        for tile in np.array_split(row_band, 3, axis=1):
            if tile.size:
                tile_backgrounds.append(float(np.percentile(tile, 85)))
    light_median = float(np.median(tile_backgrounds))
    light_tile_std = float(np.std(tile_backgrounds))
    too_dark = light_median < CAPTURE_LIGHT_MIN
    glare_fraction = float(np.mean(roi >= 252))
    glare = (light_median > CAPTURE_LIGHT_MAX
             or (glare_fraction > 0.28 and light_tile_std > 18.0))
    focus_ok = (min(band_laps) >= CAPTURE_FOCUS_BAND_MIN
                and line_lap_p20 >= CAPTURE_LINE_FOCUS_MIN)
    lighting_ok = (not too_dark and not glare
                   and light_tile_std <= CAPTURE_LIGHT_TILE_STD_MAX)
    return {
        "focus_ok": focus_ok, "lighting_ok": lighting_ok,
        "too_dark": too_dark, "glare": glare,
        "band_lap_min": min(band_laps), "band_laps": band_laps,
        "line_lap_p20": line_lap_p20,
        "light_median": light_median,
        "light_tile_std": light_tile_std,
        "roi_rect": (xa, ya, xb, yb),
    }


def _outside_ocr_text_evidence(gray, text_rect):
    """Find dark text-like strokes that OCR missed below its last box.

    This deliberately inspects the original-resolution image.  Otherwise a
    soft final paragraph can disappear during the 960px detector resize and
    the focus ROI would stop above the exact area that needs checking.
    """
    h, w = gray.shape
    x1, y1, x2, y2 = [int(value) for value in text_rect]
    inset = max(3, int(round(0.04 * max(1, x2 - x1))))
    xa, xb = max(0, x1 + inset), min(w, x2 - inset)
    ya = min(h, y2 + 3)
    yb = min(h, y2 + int(round(0.24 * h)))
    band = gray[ya:yb, xa:xb]
    empty = {
        "height": int(band.shape[0]) if band.ndim == 2 else 0,
        "mean": 0.0,
        "fraction": 0.0,
        "focus_lap": 0.0,
        "last_text_y": int(y2),
        "present": False,
    }
    if band.size == 0 or min(band.shape) < 5:
        return empty

    # Black-hat isolates thin dark strokes on light paper without adding blur
    # to the OCR image.  Kernel dimensions scale with the captured text band.
    kernel_width = max(15, (band.shape[1] // 28) | 1)
    kernel_height = max(3, (int(round(h / 154.0))) | 1)
    blackhat = cv2.morphologyEx(
        band, cv2.MORPH_BLACKHAT,
        np.ones((kernel_height, kernel_width), np.uint8))
    mean_value = float(np.mean(blackhat))
    stroke_mask = blackhat >= 12
    stroke_fraction = float(np.mean(stroke_mask))
    focus_lap = float(cv2.Laplacian(band, cv2.CV_64F).var())
    present = (
        mean_value >= CAPTURE_OUTSIDE_TEXT_MEAN_MIN
        and stroke_fraction >= CAPTURE_OUTSIDE_TEXT_FRACTION_MIN
    )
    last_text_y = int(y2)
    if present:
        active_rows = np.flatnonzero(np.mean(stroke_mask, axis=1) >= 0.025)
        if len(active_rows):
            # Ignore short paper-edge/shadow runs at the end of the search
            # band. Real missed text occupies at least roughly two black-hat
            # kernel heights, even when it is only one printed line.
            runs = []
            run_start = run_end = int(active_rows[0])
            for row_index in active_rows[1:]:
                row_index = int(row_index)
                if row_index > run_end + 1:
                    runs.append((run_start, run_end))
                    run_start = row_index
                run_end = row_index
            runs.append((run_start, run_end))
            minimum_run = max(3, 2 * kernel_height)
            text_runs = [
                (start, end) for start, end in runs
                if end - start + 1 >= minimum_run
            ]
            if text_runs:
                last_text_y = min(h, ya + text_runs[-1][1] + 1)
    return {
        "height": int(band.shape[0]),
        "mean": mean_value,
        "fraction": stroke_fraction,
        "focus_lap": focus_lap,
        "last_text_y": last_text_y,
        "present": present,
    }


def analyze_capture_frame(frame, ocr_engine):
    """Analyze one immutable frame; all returned geometry belongs to it."""
    h_orig, w_orig = frame.shape[:2]
    scale = min(1.0, CAPTURE_ANALYSIS_WIDTH / float(w_orig))
    work = cv2.resize(
        frame, (int(round(w_orig * scale)), int(round(h_orig * scale))),
        interpolation=cv2.INTER_AREA)
    h, w = work.shape[:2]
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    global_lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    global_brightness = float(np.mean(gray))

    started = time.perf_counter()
    with OCR_LOCK:
        ocr_result = ocr_engine.predict(work)
    detection_seconds = time.perf_counter() - started
    all_boxes = _extract_detection_boxes(ocr_result, work)
    cluster = _dominant_document_cluster(all_boxes, w, h)

    base = {
        "analysis_scale": scale,
        "analysis_size": (w, h),
        "detection_seconds": detection_seconds,
        "all_box_count": len(all_boxes),
        "global_lap": global_lap,
        "global_brightness": global_brightness,
        "page_found": False,
        "page_complete": False,
        "content_complete": False,
        "dense_content_complete": False,
        "sparse_physical_complete": False,
        "physical_page_safe": False,
        "distance_ok": False,
        "text_readable": False,
        "missing_sides": [],
        "text_edge_sides": [],
        "physical_sides": [],
        "hard_text_edge_sides": [],
        "unreadable_sides": [],
        "outside_text_below": {
            "height": 0, "mean": 0.0, "fraction": 0.0,
            "focus_lap": 0.0, "last_text_y": 0, "present": False},
        "boxes": [],
        "outer_box": None,
        "paper": None,
        "row_count": 0,
        "focus_ok": global_lap >= 75.0,
        "lighting_ok": 45.0 <= global_brightness <= 215.0,
        "too_dark": global_brightness < 45.0,
        "glare": False,
        "band_lap_min": global_lap,
        "line_lap_p20": 0.0,
        "light_median": global_brightness,
        "light_tile_std": 0.0,
        "geometry_center": None,
        "geometry_area": 0.0,
        "geometry_angle": 0.0,
    }
    if cluster is None or len(cluster["boxes"]) < 2:
        return base

    quality = _document_quality(work, cluster)
    # The paper component remains useful telemetry, but it is deliberately not
    # part of capture eligibility or spoken direction in Box4b.
    paper = _strict_paper_component(work, cluster["rect"])
    x1, y1, x2, y2 = cluster["rect"]
    inv_scale = 1.0 / scale
    detected_rect_orig = (
        max(0, int(x1 * inv_scale)), max(0, int(y1 * inv_scale)),
        min(w_orig, int(x2 * inv_scale)), min(h_orig, int(y2 * inv_scale)),
    )
    original_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    outside_text_below = _outside_ocr_text_evidence(
        original_gray, detected_rect_orig)
    if outside_text_below["present"]:
        # The 960px live detector can miss a sharp final paragraph. Extend the
        # text envelope to the visible strokes instead of calling the page cut.
        y2 = min(
            float(h),
            max(y2, outside_text_below["last_text_y"] * scale))
    unreadable_sides = (
        ["B"] if (outside_text_below["present"]
                  and outside_text_below["focus_lap"]
                  < CAPTURE_EXTRA_TEXT_FOCUS_MIN) else [])

    median_height = max(1.0, cluster["median_height"])
    margin_pixels = {
        "L": max(0.0, x1), "T": max(0.0, y1),
        "R": max(0.0, w - x2), "B": max(0.0, h - y2),
    }
    text_margins = {
        "L": margin_pixels["L"] / w,
        "T": margin_pixels["T"] / h,
        "R": margin_pixels["R"] / w,
        "B": margin_pixels["B"] / h,
    }
    text_clearance_lines = {
        side: margin_pixels[side] / median_height for side in "LTRB"
    }
    text_edge_sides = [
        side for side in "LTRB"
        if (margin_pixels[side] <= 2.0
            or text_clearance_lines[side] < CAPTURE_TEXT_CLEARANCE_LINES)
    ]

    # Paper-edge contact is retained only so the recordings can compare it to
    # the text decision. It can never veto capture or generate a direction.
    physical_sides = []
    if paper is not None:
        for side in "LTRB":
            if (paper["fractions"][side] >= CAPTURE_PAPER_BORDER_FRACTION
                    and paper["runs"][side] >= CAPTURE_PAPER_BORDER_RUN):
                physical_sides.append(side)

    row_count = cluster["row_count"]
    page_found = len(cluster["boxes"]) >= 3
    text_envelope_complete = page_found and not text_edge_sides
    content_complete = text_envelope_complete
    page_complete = text_envelope_complete
    dense_content_complete = row_count > 12 and text_envelope_complete
    sparse_physical_complete = row_count <= 12 and text_envelope_complete
    physical_page_safe = paper is not None and not physical_sides
    missing_sides = list(text_edge_sides) if not text_envelope_complete else []

    geometry_width = max(1.0, x2 - x1)
    geometry_height = max(1.0, y2 - y1)
    cluster_width_ratio = geometry_width / w
    page_long_ratio = 0.0
    if paper is not None:
        _, _, page_width, page_height = paper["bbox"]
        page_long_ratio = max(page_width / w, page_height / h)
    median_line_ratio = median_height / h
    too_far = page_found and median_line_ratio < CAPTURE_MIN_LINE_HEIGHT_RATIO
    distance_ok = page_complete and not too_far

    rect_orig = (
        max(0, int(x1 * inv_scale)), max(0, int(y1 * inv_scale)),
        min(w_orig, int(x2 * inv_scale)), min(h_orig, int(y2 * inv_scale)),
    )

    boxes_orig = [
        (box["poly"] * inv_scale).astype(np.int32)
        for box in cluster["boxes"]
    ]
    paper_hull_orig = None
    if paper is not None:
        paper_hull_orig = (paper["hull"].astype(np.float32) * inv_scale).astype(np.int32)
    outer_box = {
        "rect": rect_orig,
        "touching_edge": not page_complete,
        "missing_sides": list(missing_sides),
        "paper_hull": paper_hull_orig,
        "text_margins": text_margins,
    }
    base.update(quality)
    base.update({
        "page_found": page_found,
        "page_complete": page_complete,
        "content_complete": content_complete,
        "text_envelope_complete": text_envelope_complete,
        "dense_content_complete": dense_content_complete,
        "sparse_physical_complete": sparse_physical_complete,
        "physical_page_safe": physical_page_safe,
        "distance_ok": distance_ok,
        "too_far": too_far,
        "text_readable": page_found and not unreadable_sides,
        "missing_sides": missing_sides,
        "text_edge_sides": text_edge_sides,
        "physical_sides": physical_sides,
        "hard_text_edge_sides": text_edge_sides,
        "unreadable_sides": unreadable_sides,
        "outside_text_below": outside_text_below,
        "boxes": boxes_orig,
        "outer_box": outer_box,
        "paper": paper,
        "row_count": row_count,
        "cluster_box_count": len(cluster["boxes"]),
        "cluster_width_ratio": cluster_width_ratio,
        "cluster_height_ratio": geometry_height / h,
        "cluster_background": cluster["background"],
        "max_gap_ratio": cluster["max_gap_ratio"],
        "median_line_ratio": median_line_ratio,
        "page_long_ratio": page_long_ratio,
        "text_margins": text_margins,
        "text_clearance_lines": text_clearance_lines,
        "geometry_center": ((x1 + x2) / (2.0 * w), (y1 + y2) / (2.0 * h)),
        "geometry_area": geometry_width * geometry_height / float(w * h),
        "geometry_angle": cluster["angle"],
    })
    return base


def detect_text_region_size(frame, ocr_engine):
    """Compatibility wrapper backed by Box4b's synchronized text analysis."""
    assessment = analyze_capture_frame(frame, ocr_engine)
    if assessment["missing_sides"]:
        hint = "further" if len(assessment["missing_sides"]) > 1 else "align"
    elif assessment.get("too_far"):
        hint = "closer"
    else:
        hint = None
    return (hint, assessment["boxes"], assessment["outer_box"],
            assessment["row_count"])


# ══════════════════════════════════════════════════════════════════════════════
#  NON-BLOCKING DETECTION WORKER & THREAD CONTAINER
# ══════════════════════════════════════════════════════════════════════════════
class DetectionResult:
    """Thread-safe storage for distinct, frame-synchronized analyses."""
    def __init__(self):
        self._lock = threading.Lock()
        self._payload = None
        self._generation = 0
        self._running = False

    def update(self, assessment, frame, sequence, frame_timestamp, error=None):
        with self._lock:
            self._generation += 1
            self._payload = {
                "generation": self._generation,
                "assessment": assessment,
                "frame": frame,
                "sequence": sequence,
                "frame_timestamp": frame_timestamp,
                "completed_at": time.monotonic(),
                "error": error,
            }
            self._running = False

    def get(self):
        with self._lock:
            return self._payload, self._running

    def set_running(self):
        with self._lock:
            if self._running:
                return False
            self._running = True
            return True

    def is_running(self):
        with self._lock:
            return self._running


def _detection_worker(frame, sequence, frame_timestamp, ocr_engine, det_result):
    try:
        assessment = analyze_capture_frame(frame, ocr_engine)
        det_result.update(
            assessment, frame, sequence, frame_timestamp, error=None)
    except Exception as exc:
        print(f"  [CAPTURE ANALYSIS] Failed: {exc}")
        det_result.update(
            None, frame, sequence, frame_timestamp, error=str(exc))


# ══════════════════════════════════════════════════════════════════════════════
#  QUALITY OVERLAY ON PREVIEW (with outer page box)
# ══════════════════════════════════════════════════════════════════════════════
def draw_quality_overlay(frame, assessment=None, stability_count=0):
    """Draw only the dominant document cluster and Box4b gate decisions."""
    display = frame.copy()
    h, w = display.shape[:2]
    if assessment is not None:
        for poly in assessment.get("boxes", []):
            points = poly.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(
                display, [points], isClosed=True,
                color=(255, 255, 0), thickness=2)
        outer = assessment.get("outer_box")
        if outer is not None:
            x1, y1, x2, y2 = outer["rect"]
            text_ok = bool(assessment.get("text_envelope_complete"))
            box_color = (0, 220, 0) if text_ok else (0, 0, 230)
            sides = "".join(assessment.get("missing_sides", []))
            unreadable = "".join(assessment.get("unreadable_sides", []))
            label = "TEXT SAFE" if text_ok else (
                f"TEXT SOFT: {unreadable}" if unreadable else
                f"TEXT EDGE: {sides}" if sides else "TEXT NOT CONFIRMED")
            cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 3)
            cv2.putText(
                display, label, (x1 + 5, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, box_color, 2, cv2.LINE_AA)

    bar_h = 68
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
    if assessment is None:
        status = "Analyzing page ..."
        details = "Waiting for a fresh OCR result"
        color = (0, 200, 255)
    else:
        critical_ok = all((
            assessment.get("analysis_fresh", True),
            assessment.get("page_complete", False),
            assessment.get("temporal_coverage_ok", True),
            assessment.get("distance_ok", False),
            assessment.get("focus_ok", False),
            assessment.get("lighting_ok", False),
            assessment.get("motion_ok", False),
        ))
        color = (0, 220, 0) if critical_ok else (0, 0, 230)
        status = assessment.get("guidance_text", "Checking page")
        reference_rows = (assessment.get("coverage_reference_rows")
                          or assessment.get("row_count", 0))
        details = (
            f"Rows:{assessment.get('row_count', 0)}/{reference_rows:.0f}  "
            f"BandLap:{assessment.get('band_lap_min', 0):.0f}  "
            f"LineLap:{assessment.get('line_lap_p20', 0):.0f}  "
            f"LightStd:{assessment.get('light_tile_std', 0):.1f}  "
            f"Stable:{stability_count}/{CAPTURE_REQUIRED_DISTINCT}")
    cv2.putText(
        display, status, (12, 27), cv2.FONT_HERSHEY_SIMPLEX,
        0.62, color, 2, cv2.LINE_AA)
    cv2.putText(
        display, details, (12, 55), cv2.FONT_HERSHEY_SIMPLEX,
        0.46, (235, 235, 235), 1, cv2.LINE_AA)
    return display


# ══════════════════════════════════════════════════════════════════════════════
#  PRE-CAPTURE QUALITY LOOP
# ══════════════════════════════════════════════════════════════════════════════
def _estimate_capture_motion(previous_payload, current_payload):
    """Measure camera/page movement between two independently analyzed frames."""
    if previous_payload is None:
        return {"known": False, "ok": True, "score": 0.0,
                "shift": 0.0, "geometry_shift": 0.0}
    previous = previous_payload["frame"]
    current = current_payload["frame"]
    previous_gray = cv2.cvtColor(
        cv2.resize(previous, (320, 180), interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2GRAY).astype(np.float32)
    current_gray = cv2.cvtColor(
        cv2.resize(current, (320, 180), interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2GRAY).astype(np.float32)
    try:
        (shift_x, shift_y), response = cv2.phaseCorrelate(
            previous_gray, current_gray)
        shift_ratio = float(np.hypot(shift_x / 320.0, shift_y / 180.0))
        if response < 0.04:
            shift_ratio = max(
                shift_ratio,
                float(np.mean(np.abs(previous_gray - current_gray))) / 255.0)
    except cv2.error:
        shift_ratio = float(np.mean(
            np.abs(previous_gray - current_gray))) / 255.0

    previous_assessment = previous_payload.get("assessment") or {}
    current_assessment = current_payload.get("assessment") or {}
    previous_center = previous_assessment.get("geometry_center")
    current_center = current_assessment.get("geometry_center")
    geometry_shift = 0.0
    area_change = 0.0
    angle_change = 0.0
    if previous_center is not None and current_center is not None:
        geometry_shift = float(np.hypot(
            previous_center[0] - current_center[0],
            previous_center[1] - current_center[1]))
        previous_area = max(previous_assessment.get("geometry_area", 0.0), 1e-6)
        current_area = max(current_assessment.get("geometry_area", 0.0), 1e-6)
        area_change = abs(current_area / previous_area - 1.0)
        angle_change = abs(
            current_assessment.get("geometry_angle", 0.0)
            - previous_assessment.get("geometry_angle", 0.0))
    diagnostic_score = max(
        shift_ratio / CAPTURE_MOTION_SHIFT_MAX,
        geometry_shift / CAPTURE_MOTION_GEOMETRY_MAX,
        area_change / CAPTURE_MOTION_AREA_CHANGE_MAX,
        angle_change / CAPTURE_MOTION_ANGLE_MAX,
    )
    # OCR polygons appear/disappear between otherwise sharp frames. Their
    # centre/area/angle remain logged, but only direct image motion controls the
    # gate; actual motion blur is independently rejected by the focus checks.
    score = shift_ratio / CAPTURE_MOTION_SHIFT_MAX
    return {
        "known": True,
        "ok": shift_ratio <= CAPTURE_MOTION_SHIFT_MAX,
        "score": float(score),
        "diagnostic_score": float(diagnostic_score),
        "shift": shift_ratio,
        "geometry_shift": geometry_shift,
        "area_change": area_change,
        "angle_change": angle_change,
    }


def _guidance_for_assessment(assessment):
    """Choose one truthful instruction using explicit failure-state priority."""
    motion_known = assessment.get("motion_known", False)
    motion_ok = assessment.get("motion_ok", True)
    if not assessment.get("analysis_fresh", True):
        return "checking"
    if not assessment.get("page_found", False):
        if assessment.get("too_dark", False):
            return "too_dark"
        if assessment.get("glare", False):
            return "glare"
        if (assessment.get("all_box_count", 0) >= 2
                and not assessment.get("focus_ok", False)):
            return "blurry"
        if motion_known and not motion_ok:
            return "hold_still"
        return "page_not_found"

    if assessment.get("too_dark", False):
        return "too_dark"
    if assessment.get("glare", False):
        return "glare"
    if not assessment.get("lighting_ok", False):
        return "bad_lighting"
    if (assessment.get("unreadable_sides")
            or not assessment.get("focus_ok", False)):
        return "blurry"
    if motion_known and not motion_ok:
        return "hold_still"
    if not assessment.get("page_complete", False):
        # Missing sides come only from the text envelope in Box4b. Physical A4
        # borders and an unknown classifier result can never generate a turn.
        sides = assessment.get("missing_sides", [])
        if len(sides) > 1:
            return "move_back"
        if sides:
            return {
                "T": "look_up", "B": "look_down",
                "L": "look_left", "R": "look_right",
            }[sides[0]]
        return "page_not_found"
    if not assessment.get("temporal_coverage_ok", True):
        return "hold_still"
    if not assessment.get("distance_ok", False):
        return "move_closer"
    return "almost_ready"


def _capture_gate_passes(assessment, already_stable=False):
    """Independent hard gates; lighting can never compensate for bad focus."""
    focus_ok = assessment.get("focus_ok", False)
    lighting_ok = assessment.get("lighting_ok", False)
    if already_stable:
        # Small hysteresis prevents a one-point threshold flutter while still
        # requiring every accepted observation to be safely usable.
        focus_ok = (
            assessment.get("band_lap_min", 0.0) >= 0.88 * CAPTURE_FOCUS_BAND_MIN
            and assessment.get("line_lap_p20", 0.0) >= 0.88 * CAPTURE_LINE_FOCUS_MIN)
        lighting_ok = (
            not assessment.get("too_dark", False)
            and not assessment.get("glare", False)
            and assessment.get("light_tile_std", 999.0)
            <= CAPTURE_LIGHT_TILE_STD_MAX + 3.0)
    return all((
        assessment.get("analysis_fresh", True),
        assessment.get("page_complete", False),
        assessment.get("temporal_coverage_ok", True),
        assessment.get("distance_ok", False),
        assessment.get("text_readable", False),
        focus_ok,
        lighting_ok,
        assessment.get("motion_ok", True),
    ))


def pre_capture_quality_loop(frame_holder, ocr_engine, tts_module,
                             key_queue, event_queue, debug_recorder=None):
    """Guide a head-mounted camera and capture three distinct safe frames."""
    print("\n[QUALITY] Box4b text-envelope capture guidance active …")
    print("[QUALITY] Checking all printed text, focus, lighting, and image motion.")
    print("[QUALITY] Three good frames in the latest four are required. S force-captures.")

    guidance = None
    try:
        guidance = AudioGuidance(
            tts_module, state_cooldown=4.5, global_cooldown=0.80,
            event_callback=(
                lambda event_type, **details: _safe_debug_call(
                    debug_recorder, "record_event", event_type, **details)
                if debug_recorder is not None else None))
    except Exception as exc:
        print(f"[GUIDANCE] Spoken guidance unavailable: {exc}")

    detection_result = DetectionResult()
    last_analysis_started = 0.0
    last_started_sequence = -1
    last_processed_generation = 0
    previous_payload = None
    latest_assessment = None
    latest_assessment_timestamp = 0.0
    stability_window = []
    stable_payloads = []
    coverage_history = []
    pending_guidance = None
    pending_guidance_count = 0

    try:
        while True:
            frame, sequence, frame_timestamp = frame_holder.get_snapshot()
            if frame is None:
                time.sleep(0.08)
                continue
            now = time.monotonic()

            if (sequence != last_started_sequence
                    and now - last_analysis_started >= CAPTURE_ANALYSIS_INTERVAL
                    and detection_result.set_running()):
                last_started_sequence = sequence
                last_analysis_started = now
                threading.Thread(
                    target=_detection_worker,
                    args=(frame, sequence, frame_timestamp,
                          ocr_engine, detection_result),
                    daemon=True).start()

            payload, _ = detection_result.get()
            if (payload is not None
                    and payload["generation"] != last_processed_generation):
                last_processed_generation = payload["generation"]
                assessment = payload.get("assessment")
                passed = False
                motion = {}
                allow_guidance = False
                if assessment is None:
                    latest_assessment = None
                    latest_assessment_timestamp = 0.0
                    stability_window = []
                    stable_payloads = []
                    state = "page_not_found"
                    print(f"  [ANALYSIS] Error: {payload.get('error')}")
                    _safe_debug_call(
                        debug_recorder, "record_event",
                        "capture_analysis_failed",
                        generation=payload.get("generation"),
                        frame_sequence=payload.get("sequence"),
                        error=payload.get("error"))
                else:
                    analysis_age = max(
                        0.0, time.monotonic() - payload["frame_timestamp"])
                    analysis_fresh = analysis_age <= CAPTURE_MAX_ANALYSIS_AGE
                    assessment["analysis_age"] = analysis_age
                    assessment["analysis_fresh"] = analysis_fresh

                    if (analysis_fresh
                            and assessment.get("page_found", False)
                            and assessment.get("focus_ok", False)
                            and assessment.get("lighting_ok", False)):
                        coverage_history.append(
                            float(assessment.get("row_count", 0)))
                        coverage_history = coverage_history[-CAPTURE_COVERAGE_HISTORY:]
                    reference_rows = (
                        float(np.percentile(coverage_history, 90))
                        if coverage_history else
                        float(assessment.get("row_count", 0)))
                    temporal_coverage_ok = (
                        len(coverage_history) < 4
                        or assessment.get("row_count", 0)
                        >= CAPTURE_COVERAGE_RATIO * max(reference_rows, 1.0))
                    assessment["coverage_reference_rows"] = reference_rows
                    assessment["temporal_coverage_ok"] = temporal_coverage_ok

                    motion = _estimate_capture_motion(previous_payload, payload)
                    assessment["motion_known"] = motion["known"]
                    assessment["motion_ok"] = motion["ok"]
                    assessment["motion_score"] = motion["score"]
                    state = _guidance_for_assessment(assessment)
                    assessment["guidance_state"] = state
                    assessment["guidance_text"] = GUIDANCE_PROMPTS[state]
                    latest_assessment = assessment if analysis_fresh else None
                    latest_assessment_timestamp = (
                        payload["frame_timestamp"] if analysis_fresh else 0.0)
                    allow_guidance = analysis_fresh and state != "checking"

                    passed = _capture_gate_passes(
                        assessment, already_stable=bool(stable_payloads))
                    hard_failure = bool(
                        analysis_fresh and (
                            not assessment.get("page_found", False)
                            or assessment.get("missing_sides")
                            or assessment.get("unreadable_sides")))
                    previous_pass_count = len(stable_payloads)
                    if hard_failure:
                        stability_window = []
                    else:
                        stability_window.append((payload, passed))
                        stability_window = stability_window[-CAPTURE_STABILITY_WINDOW:]
                    stable_payloads = [
                        item for item, item_passed in stability_window
                        if item_passed]
                    if (not passed and previous_pass_count
                            and not hard_failure):
                        print(
                            "  [STABLE HOLD] One soft failure retained; "
                            f"{len(stable_payloads)}/{CAPTURE_REQUIRED_DISTINCT} "
                            "good frames remain.")
                    elif hard_failure and previous_pass_count:
                        print(
                            "  [STABLE HOLD] Cleared by a confirmed text-edge "
                            "or unreadable-text failure.")

                    sides = "".join(assessment.get("missing_sides", [])) or "none"
                    print(
                        f"  [OBS {payload['generation']:03d}] "
                        f"state={state}, rows={assessment.get('row_count', 0)}/"
                        f"{reference_rows:.0f}, age={analysis_age:.2f}s, "
                        f"edges={sides}, bandLap={assessment.get('band_lap_min', 0):.0f}, "
                        f"lineLap={assessment.get('line_lap_p20', 0):.0f}, "
                        f"lightStd={assessment.get('light_tile_std', 0):.1f}, "
                        f"belowText={assessment.get('outside_text_below', {}).get('mean', 0):.1f}, "
                        f"motion={assessment.get('motion_score', 0):.2f}, "
                        f"stable={len(stable_payloads)}/{CAPTURE_REQUIRED_DISTINCT}")

                    _safe_debug_call(
                        debug_recorder, "record_analysis",
                        payload, assessment, state, passed,
                        len(stable_payloads), motion=motion)

                    if (passed
                            and len(stable_payloads) >= CAPTURE_REQUIRED_DISTINCT
                            and (stable_payloads[-1]["completed_at"]
                                 - stable_payloads[0]["completed_at"]
                                 >= CAPTURE_MIN_STABLE_SECONDS)):
                        chosen = max(
                            stable_payloads,
                            key=lambda item: (
                                item["assessment"].get("row_count", 0),
                                item["assessment"].get("cluster_box_count", 0),
                                item["assessment"].get("band_lap_min", 0.0)
                                + 0.25 * item["assessment"].get(
                                    "line_lap_p20", 0.0),
                                item.get("completed_at", 0.0)))
                        print(
                            "\n[QUALITY] ✅ Three of the latest four observations passed; "
                            f"capturing analyzed frame #{chosen['sequence']}.")
                        _safe_debug_call(
                            debug_recorder, "record_event",
                            "automatic_capture",
                            generation=chosen.get("generation"),
                            frame_sequence=chosen.get("sequence"),
                            stability_count=len(stable_payloads))
                        beep_ready()
                        return chosen["frame"]
                    if analysis_fresh:
                        previous_payload = payload

                if allow_guidance:
                    if state == pending_guidance:
                        pending_guidance_count += 1
                    else:
                        pending_guidance = state
                        pending_guidance_count = 1
                    directional_states = {
                        "look_up", "look_down", "look_left", "look_right",
                        "move_back", "move_closer",
                    }
                    confirmation_count = (
                        3 if state in directional_states else 2)
                    if (guidance is not None
                            and pending_guidance_count >= confirmation_count):
                        guidance.notify(state)

            display_assessment = latest_assessment
            if (latest_assessment_timestamp <= 0.0
                    or time.monotonic() - latest_assessment_timestamp
                    > CAPTURE_MAX_ANALYSIS_AGE):
                # Never draw old boxes/directions on a newer live pose.
                display_assessment = None
            display = draw_quality_overlay(
                frame, display_assessment, len(stable_payloads))
            cv2.imshow(
                "Smart Glasses Live Stream", cv2.resize(display, (640, 360)))
            cv2.waitKey(1)

            try:
                while not key_queue.empty():
                    key = key_queue.get_nowait()
                    if key == 'q':
                        print("[QUALITY] ❌ Cancelled by user (Q)")
                        _safe_debug_call(
                            debug_recorder, "record_event",
                            "capture_cancelled", source="keyboard")
                        return None
                    if key == 's':
                        print(
                            "[QUALITY] ⚠ FORCE-CAPTURE from keyboard: "
                            "all Box4b safety gates were bypassed.")
                        _safe_debug_call(
                            debug_recorder, "record_event",
                            "forced_capture", source="keyboard",
                            frame_sequence=sequence)
                        beep_ready()
                        return frame
            except queue.Empty:
                pass

            try:
                while not event_queue.empty():
                    event = event_queue.get_nowait()
                    if event.get("event") == "disconnect":
                        _safe_debug_call(
                            debug_recorder, "record_event",
                            "capture_cancelled", source="disconnect")
                        return None
                    if event.get("cmd") == "capture_from_pi":
                        print(
                            "[QUALITY] ⚠ FORCE-CAPTURE from Pi: "
                            "all Box4b safety gates were bypassed.")
                        _safe_debug_call(
                            debug_recorder, "record_event",
                            "forced_capture", source="pi",
                            frame_sequence=sequence)
                        beep_ready()
                        return frame
            except queue.Empty:
                pass
            time.sleep(0.025)
    finally:
        if guidance is not None:
            guidance.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  PROCESS & SPEAK
# ══════════════════════════════════════════════════════════════════════════════
def process_and_speak(frame, run_count, language, book_mode,
                      ocr_engine, unwarper, translator, tokenizer, tts_module,
                      key_queue=None, event_queue=None):
    print(f"\n{'─'*60}")
    print(f"  RUN #{run_count}")
    print(f"{'─'*60}\n")

    pipeline_t0 = time.time()
    beep_capture()
    img_path = save_frame(frame, run_count)

    # Score the actual JPEG that OCR would consume, not only the live frame.
    saved_frame = cv2.imread(img_path)
    if saved_frame is None:
        print("[POST-CAPTURE] Could not reload saved image; retry requested.")
        beep_failure()
        return False
    _, post_details = score_frame_quality(saved_frame)
    print(
        f"[POST-CAPTURE] LapVar={post_details['sharpness']:.1f}, "
        f"score components: sharp={post_details['sharpness_pts']}, "
        f"bright={post_details['brightness_pts']}, "
        f"even={post_details['evenness_pts']}")

    cv2.imshow("Captured Image (OCR Target)", cv2.resize(frame, (640, 360)))
    cv2.waitKey(1)

    # Stage 1: OCR with line coordinates and paragraph grouping.
    extracted_text, paragraphs, ocr_image, t_ocr = run_ocr(
        ocr_engine, img_path, book_mode, unwarper)
    if not extracted_text or not paragraphs:
        tone_failure()
        tts_module.speak("No text detected. Please try again.")
        tts_module.wait_until_done()
        print("[WARN] No text extracted. Skipping.\n")
        return False

    tone_success()
    if ocr_image is None:
        ocr_image = saved_frame
    show_paragraph_preview(ocr_image, paragraphs, active_index=None)

    print(f"\n{'='*60}")
    print("  EXTRACTED TEXT")
    print(f"{'='*60}")
    print(extracted_text)
    print(f"{'='*60}\n")

    counts = _region_counts(paragraphs)
    print(
        f"[REGIONS] Detected {len(paragraphs)} text region(s): "
        f"{counts['titles']} title, {counts['headings']} heading(s), "
        f"{counts['paragraphs']} paragraph(s)")
    for paragraph in paragraphs:
        location = paragraph_page_location(paragraph, ocr_image.shape)
        print(
            f"  {_region_label(paragraph)} | "
            f"type={paragraph.get('region_type', 'paragraph')} | "
            f"{len(paragraph['lines'])} line(s) | "
            f"confidence={paragraph['confidence']:.3f} | {location}")
        print(f"    {paragraph['source_text']}")

    # Stage 2: translate all paragraph chunks in one GPU batch, retaining IDs.
    translated_paragraphs, t_translate = run_translation_nllb_paragraphs(
        paragraphs, language, translator, tokenizer)
    english_text = " ".join(
        paragraph["translated_text"] for paragraph in translated_paragraphs)

    if language != "english":
        print(f"\n{'='*60}")
        print("  CLASSIFIED REGION TRANSLATIONS (English) — NLLB 1.3B")
        print(f"{'='*60}")
        for paragraph in translated_paragraphs:
            print(
                f"[{_region_label(paragraph)}] "
                f"{paragraph['translated_text']}")
        print(f"{'='*60}\n")

    # Stage 2.5: apply the existing conservative SymSpell rules separately.
    t_spell = 0.0
    all_spell_changes = []
    spell_t0 = time.time()
    for paragraph in translated_paragraphs:
        translated_text = paragraph["translated_text"]
        if SPELL_CORRECTOR is not None:
            corrected_text, changes = run_spell_correction(
                SPELL_CORRECTOR, translated_text)
        else:
            corrected_text, changes = translated_text, []
        paragraph["spoken_text"] = corrected_text
        paragraph["spell_changes"] = changes
        for original, replacement in changes:
            all_spell_changes.append(
                (_region_label(paragraph), original, replacement))
    t_spell = time.time() - spell_t0 if SPELL_CORRECTOR is not None else 0.0
    corrected_text = " ".join(
        paragraph["spoken_text"] for paragraph in translated_paragraphs)

    if SPELL_CORRECTOR is not None:
        if all_spell_changes:
            print(
                f"[STAGE 2.5] Applied {len(all_spell_changes)} conservative "
                f"correction(s) in {t_spell:.3f}s:")
            for region_label, original, replacement in all_spell_changes:
                print(
                    f"  {region_label}: {original} -> {replacement}")
        else:
            print(
                f"[STAGE 2.5] No sufficiently confident corrections; "
                f"original preserved ({t_spell:.3f}s)")

    # DEBUG — remove when done
    save_run_debug_outputs(
        run_count, img_path, ocr_image, translated_paragraphs,
        extracted_text, language, all_spell_changes)

    # Save both full-page text and the source-to-audio paragraph mapping.
    try:
        log_dir = os.path.join(PROJECT_DIR, "1Pipeline", "final", "working")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "ocr_results_comparison.txt")
        with open(log_path, "a", encoding="utf-8") as f_log:
            f_log.write("="*80 + "\n")
            f_log.write(f"TIMESTAMP   : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f_log.write("CODE VERSION: pipeline_cli_box4b.py\n")
            f_log.write(f"RUN NUMBER  : {run_count}\n")
            f_log.write(f"IMAGE FILE  : {os.path.basename(img_path)}\n")
            log_counts = _region_counts(translated_paragraphs)
            f_log.write(f"TEXT REGIONS: {len(translated_paragraphs)}\n")
            f_log.write(f"PAGE TITLES : {log_counts['titles']}\n")
            f_log.write(f"HEADINGS    : {log_counts['headings']}\n")
            f_log.write(f"PARAGRAPHS  : {log_counts['paragraphs']}\n")
            f_log.write("-"*80 + "\n")
            f_log.write("EXTRACTED TEXT:\n")
            f_log.write(extracted_text + "\n")
            if language != "english":
                f_log.write("-"*80 + "\n")
                f_log.write("TRANSLATED TEXT:\n")
                f_log.write(english_text + "\n")
            if all_spell_changes:
                f_log.write("-"*80 + "\n")
                f_log.write("CONSERVATIVELY CORRECTED TEXT:\n")
                f_log.write(corrected_text + "\n")

            for paragraph in translated_paragraphs:
                f_log.write("-"*80 + "\n")
                f_log.write(
                    f"REGION {_region_label(paragraph)} | "
                    f"ID={paragraph.get('region_id', paragraph['number'])} | "
                    f"TYPE={paragraph.get('region_type', 'paragraph')} | "
                    f"BBOX={paragraph['bbox']} | "
                    f"CONFIDENCE={paragraph['confidence']:.3f}\n")
                f_log.write(
                    f"SOURCE     : {paragraph['source_text']}\n")
                if language != "english":
                    f_log.write(
                        f"TRANSLATED : {paragraph['translated_text']}\n")
                f_log.write(
                    f"SPOKEN     : {paragraph['spoken_text']}\n")
                if paragraph["spell_changes"]:
                    f_log.write(
                        f"CORRECTIONS: {paragraph['spell_changes']}\n")
            f_log.write("="*80 + "\n\n")
    except Exception as log_error:
        print(f"[LOG ERROR] Failed to save OCR output to log: {log_error}")

    # Stage 3: each spoken title/heading/paragraph stays tied to its source box.
    t_tts = run_tts_paragraphs(
        tts_module, translated_paragraphs, ocr_image,
        key_queue, event_queue)

    total_pipeline = time.time() - pipeline_t0
    print(f"\n{'─'*40}")
    print("  ⏱  TIMING BREAKDOWN")
    print(f"{'─'*40}")
    print(f"  OCR          : {t_ocr:.3f}s")
    if language != "english":
        print(f"  Translation  : {t_translate:.3f}s")
    else:
        print("  Translation  : skipped")
    print(f"  Spell Corr   : {t_spell:.3f}s")
    print(f"  TTS Playback : {t_tts:.3f}s")
    print(f"{'─'*40}")
    print(f"  TOTAL        : {total_pipeline:.3f}s")
    print(f"{'─'*40}")
    print("\n  Ready for next capture!\n")
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  CLI HELPERS
# ══════════════════════════════════════════════════════════════════════════════
BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║   FYDP SMART GLASSES — BOX3D3 CLASSIFIED REGIONS          ║
║          OCR → TRANSLATE (NLLB 1.3B) → SPEAK               ║
║                                                              ║
║  Controls:                                                   ║
║      S  =  quality/capture; during TTS, stop playback       ║
║      A  =  pause/resume document TTS                        ║
║      Q  =  quit                                              ║
║  Also works with S/Q on the Pi keyboard!                     ║
║                                                              ║
║  Fixes over box3d:                                           ║
║      • Relaxed edge margin (1.5%) + % edge check             ║
║      • Indent-based paragraph detection                      ║
║      • Tighter gap ceiling for large fonts                   ║
║      • Stability Hold: requires holding still for 1s         ║
║      • Upscales 720p stream to 1.5x (1080p equivalent)       ║
╚══════════════════════════════════════════════════════════════╝
"""


class SuppressOutput:
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._stdout
        sys.stderr = self._stderr


def ask_language():
    print("\n── Select source language ──")
    print("  1) French")
    print("  2) Chinese")
    print("  3) Spanish")
    print("  4) English (no translation, OCR + TTS only)")
    while True:
        choice = input("\nEnter 1/2/3/4: ").strip()
        if choice == "1": return "french"
        if choice == "2": return "chinese"
        if choice == "3": return "spanish"
        if choice == "4": return "english"
        print("[!] Invalid choice.")


def ask_book_mode():
    print("\n── Book Mode (curved page unwarping) ──")
    while True:
        choice = input("Enable book mode? (y/n): ").strip().lower()
        if choice in ("y", "yes"): return True
        if choice in ("n", "no"):  return False
        print("[!] Enter y or n.")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def run_pipeline(debug_recorder=None):
    global SPELL_CORRECTOR
    print(BANNER)

    print("[INFO] Initializing audio subsystem …")
    with SuppressOutput():
        tts_module = load_tts()

    tts_module.speak("Welcome to F Y D P Glasses")

    tts_module.wait_until_done()

    tts_module.speak("Choose language")
    language = ask_language()

    print("[INFO] Loading language-aware OCR and unwarping models (silently) …")
    with SuppressOutput():
        ocr_engine = load_ocr_engine(language)
        unwarper   = load_unwarper()

    tts_module.speak("Choose book mode")
    book_mode = ask_book_mode()

    if not book_mode:
        unwarper = None

    print(f"\n[INFO] Loading NLLB Translator for {language.upper()} …")
    with SuppressOutput():
        translator, tokenizer = load_nllb_translator(language)

    print("[INFO] Loading conservative SymSpell dictionary …")
    try:
        SPELL_CORRECTOR = load_spell_corrector()
    except Exception as spell_error:
        SPELL_CORRECTOR = None
        print(
            f"[WARN] SymSpell unavailable; continuing with original text: "
            f"{spell_error}")

    tts_module.speak("All models loaded")
    tts_module.wait_until_done()
    try:
        # Build the short guidance audio once at startup. Capture-loop playback
        # then uses cached arrays and never blocks the document TTS queue.
        prepare_guidance_clips(tts_module)
    except Exception as guidance_error:
        print(f"[GUIDANCE] Could not pre-render prompts: {guidance_error}")

    server_sock, client_sock = start_server(PORT)
    tts_module.speak("Smart glasses connected. Press S to start translating.")
    tts_module.wait_until_done()

    frame_holder = FrameHolder()
    event_queue  = queue.Queue()
    recv_thread  = threading.Thread(
        target=receiver_loop,
        args=(client_sock, frame_holder, event_queue, debug_recorder),
        daemon=True)
    recv_thread.start()

    key_queue = queue.Queue()
    start_keyboard_listener(key_queue)

    pipeline_busy = threading.Event()

    print("\n⏳ Waiting for capture … (S = capture, Q = quit)\n")

    run_count = 0

    try:
        while True:
            if not pipeline_busy.is_set():
                live_frame = frame_holder.get()
                if live_frame is not None:
                    cv2.imshow("Smart Glasses Live Stream", cv2.resize(live_frame, (640, 360)))
                    cv2.waitKey(1)

            try:
                event = event_queue.get_nowait()
                if event.get("event") == "disconnect" or event.get("cmd") == "quit":
                    print("\n[INFO] Pi disconnected.")
                    break
                if event.get("cmd") == "capture_from_pi":
                    if not pipeline_busy.is_set():
                        pipeline_busy.set()
                        frame = pre_capture_quality_loop(
                            frame_holder, ocr_engine, tts_module,
                            key_queue, event_queue, debug_recorder)
                        if frame is not None:
                            run_count += 1
                            process_and_speak(
                                frame, run_count, language, book_mode,
                                ocr_engine, unwarper, translator, tokenizer, tts_module,
                                key_queue, event_queue)
                        pipeline_busy.clear()
            except queue.Empty:
                pass

            try:
                key = key_queue.get_nowait()
                if key == 'q':
                    print("\n🛑 Q pressed — quitting.")
                    break
                elif key == 's':
                    if not pipeline_busy.is_set():
                        print("  📸 Capture triggered from PC keyboard!")
                        pipeline_busy.set()
                        frame = pre_capture_quality_loop(
                            frame_holder, ocr_engine, tts_module,
                            key_queue, event_queue, debug_recorder)
                        if frame is not None:
                            run_count += 1
                            process_and_speak(
                                frame, run_count, language, book_mode,
                                ocr_engine, unwarper, translator, tokenizer, tts_module,
                                key_queue, event_queue)
                        pipeline_busy.clear()
            except queue.Empty:
                pass

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n[EXIT] Stopped by user.")
    finally:
        tts_module.speak("Goodbye")
        tts_module.wait_until_done()
        try: client_sock.close()
        except Exception: pass
        try: server_sock.close()
        except Exception: pass
        cv2.destroyAllWindows()
        print(f"\n[EXIT] Session ended. Total runs: {run_count}")


def main():
    debug_recorder = None
    if DEBUG_SESSION_RECORDING:
        try:
            debug_recorder = DebugSessionRecorder()
            debug_recorder.start_console_capture()
            print(
                f"[DEBUG] Recording this complete session in: "
                f"{debug_recorder.session_dir}")
        except Exception as exc:
            debug_recorder = None
            print(f"[DEBUG] Session recording unavailable: {exc}")
    try:
        run_pipeline(debug_recorder)
    finally:
        if debug_recorder is not None:
            session_dir = debug_recorder.session_dir
            print(f"[DEBUG] Finalizing session evidence: {session_dir}")
            try:
                debug_recorder.close()
                print(f"[DEBUG] Session evidence saved: {session_dir}")
            except Exception as exc:
                print(f"[DEBUG] Could not finalize all evidence: {exc}")


if __name__ == "__main__":
    main()

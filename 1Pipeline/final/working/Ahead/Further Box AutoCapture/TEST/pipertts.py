"""
pipertts.py  (Final Demo Version)
=================================
Enhanced Piper TTS with speed control (1x / 1.5x / 2x) and pause/resume.

Changes from original:
  - set_speed(rate)   → adjusts playback speed via sample rate scaling
  - pause()           → pauses playback at the next block boundary (~200ms)
  - resume()          → resumes from where it paused
  - is_paused()       → check pause state

All original API preserved:
  speak(text), announce(msg), wait_until_done(), preload(), stop(), is_speaking()

Prerequisites:
  pip install piper-tts sounddevice numpy
"""

import os
import queue
import time
import threading
import urllib.request

# ── Configuration ─────────────────────────────────────────────────────────────
_VOICE_NAME  = "en_US-lessac-medium"
# Models dir: one level up from final/ → 1Pipeline/piper_models/
_MODELS_DIR  = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "piper_models")
)
_MODEL_PATH  = os.path.join(_MODELS_DIR, f"{_VOICE_NAME}.onnx")
_CONFIG_PATH = os.path.join(_MODELS_DIR, f"{_VOICE_NAME}.onnx.json")
_BASE_URL    = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main"
    "/en/en_US/lessac/medium"
)

# Block size for playback — smaller = more responsive pause/stop
_BLOCK_MS = 200


# ── Model download ─────────────────────────────────────────────────────────────
def _show_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded / total_size * 100)
        print(f"\r  {pct:5.1f}%  ({downloaded/1_048_576:.1f} / {total_size/1_048_576:.1f} MB)",
              end="", flush=True)
    else:
        print(f"\r  Downloaded {downloaded/1_048_576:.1f} MB", end="", flush=True)


def ensure_model():
    """Download the Piper voice model if it isn't already present."""
    os.makedirs(_MODELS_DIR, exist_ok=True)
    files = [f"{_VOICE_NAME}.onnx", f"{_VOICE_NAME}.onnx.json"]
    for filename in files:
        dest = os.path.join(_MODELS_DIR, filename)
        if os.path.exists(dest):
            print(f"[OK] Already exists - skipping: {filename}")
            continue
        url = f"{_BASE_URL}/{filename}"
        print(f"\n[DL] Downloading: {filename}")
        print(f"     From: {url}")
        print(f"     To:   {dest}")
        urllib.request.urlretrieve(url, dest, reporthook=_show_progress)
        print()
        print(f"[OK] Saved ({os.path.getsize(dest)/1_048_576:.1f} MB)")
    print("\n[OK] Piper model files ready.")


# ── Voice loader (lazy, thread-safe) ──────────────────────────────────────────
_voice      = None
_voice_lock = threading.Lock()


def _get_voice():
    global _voice
    if _voice is not None:
        return _voice
    with _voice_lock:
        if _voice is not None:
            return _voice
        if not os.path.exists(_MODEL_PATH):
            ensure_model()
        print(f"[TTS] Loading voice: {_VOICE_NAME} …", flush=True)
        from piper import PiperVoice
        _voice = PiperVoice.load(_MODEL_PATH, config_path=_CONFIG_PATH)
        print("[TTS] Voice ready.", flush=True)
    return _voice


# ── Audio synthesis ───────────────────────────────────────────────────────────
def _synthesize(text: str):
    """Return a list of AudioChunk objects for the given text."""
    return list(_get_voice().synthesize(text))


# ── Playback state (module-level) ────────────────────────────────────────────
_speed_rate     = 1.0
_paused         = False
_stop_requested = False
_is_speaking    = False


def _play(chunks) -> None:
    """Play audio chunks with speed control and pause/resume support.
    Concatenates all chunks into a single continuous buffer first to avoid audio
    gaps/breaks, then plays smoothly via sounddevice and tracks playback progress.
    """
    import numpy as np
    import sounddevice as sd
    global _stop_requested, _paused, _speed_rate

    if not chunks:
        return

    # ── 1. Combine all chunks into a single continuous audio array ──
    sr = chunks[0].sample_rate
    ch = chunks[0].sample_channels

    audio_arrays = []
    for chunk in chunks:
        audio_int16 = chunk.audio_int16_array
        audio = np.array(audio_int16, dtype=np.int16).reshape(-1, ch)
        audio_f32 = audio.astype(np.float32) / 32768.0
        audio_arrays.append(audio_f32)

    combined_audio = np.concatenate(audio_arrays, axis=0)

    # Prepend 0.4 seconds of silence to prevent slow DAC/headset wake-up clipping
    silence_len = int(sr * 0.4)
    silence     = np.zeros((silence_len, ch), dtype=np.float32)
    combined_audio = np.concatenate([silence, combined_audio], axis=0)

    total_frames = len(combined_audio)
    pos = 0

    while pos < total_frames and not _stop_requested:
        if _paused:
            time.sleep(0.05)
            continue

        remaining_audio = combined_audio[pos:]
        effective_sr = int(sr * _speed_rate)

        t_start = time.time()
        sd.play(remaining_audio, samplerate=effective_sr)

        playing_frames = len(remaining_audio)
        duration = playing_frames / effective_sr

        last_speed = _speed_rate
        while time.time() - t_start < duration:
            if _stop_requested:
                sd.stop()
                break
            if _paused:
                # Paused! Calculate current position and stop playback
                elapsed = time.time() - t_start
                pos += int(elapsed * effective_sr)
                sd.stop()
                break
            if _speed_rate != last_speed:
                # Speed changed! Calculate position, stop, and restart with new speed
                elapsed = time.time() - t_start
                pos += int(elapsed * effective_sr)
                sd.stop()
                break
            time.sleep(0.02)
        else:
            # Finished normally
            pos = total_frames


# ── Background worker ──────────────────────────────────────────────────────────
_tts_queue     = queue.Queue()
_STOP_SENTINEL = object()


def _worker():
    global _is_speaking, _stop_requested
    while True:
        item = _tts_queue.get()
        if item is _STOP_SENTINEL:
            break
        _is_speaking = True
        try:
            if not _stop_requested:
                _play(_synthesize(item))
        except Exception as exc:
            print(f"[TTS] Playback error: {exc}", flush=True)
        finally:
            _is_speaking = False
            _tts_queue.task_done()


_worker_thread = threading.Thread(target=_worker, daemon=True, name="piper-tts")
_worker_thread.start()


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API  (original)
# ══════════════════════════════════════════════════════════════════════════════
def speak(text: str) -> None:
    """Enqueue text for TTS. Returns immediately; audio plays in background."""
    global _stop_requested, _paused
    _stop_requested = False
    _paused = False
    utterance = text.strip() if text and text.strip() else "No text detected."
    _tts_queue.put(utterance)


def announce(message: str) -> None:
    """Enqueue a system announcement (e.g. 'Switched to Chinese mode.')."""
    global _stop_requested, _paused
    _stop_requested = False
    _paused = False
    _tts_queue.put(message.strip())


def wait_until_done() -> None:
    """Block until all queued speech has finished playing."""
    _tts_queue.join()


def preload() -> None:
    """Pre-load the voice model now to avoid the first-call delay."""
    _get_voice()


def stop() -> None:
    """Stop active speech, clear the queue, and halt audio output immediately."""
    global _stop_requested, _paused
    _stop_requested = True
    _paused = False          # un-pause so blocked loops can exit
    import sounddevice as sd
    sd.stop()
    try:
        while True:
            _tts_queue.get_nowait()
            _tts_queue.task_done()
    except queue.Empty:
        pass


def is_speaking() -> bool:
    """Check if the TTS pipeline is currently playing or has queued items."""
    return _is_speaking or not _tts_queue.empty()


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API  (new — speed / pause)
# ══════════════════════════════════════════════════════════════════════════════
def set_speed(rate: float) -> None:
    """Set playback speed. 1.0 = normal, 1.5 = fast, 2.0 = fastest.
    Takes effect on the next audio block (~200 ms latency)."""
    global _speed_rate
    _speed_rate = max(0.5, min(3.0, float(rate)))
    print(f"[TTS] Speed set to {_speed_rate:.1f}x", flush=True)


def get_speed() -> float:
    """Return current playback speed multiplier."""
    return _speed_rate


def pause() -> None:
    """Pause playback at the next block boundary. Call resume() to continue."""
    global _paused
    _paused = True
    print("[TTS] Paused", flush=True)


def resume() -> None:
    """Resume paused playback."""
    global _paused
    _paused = False
    print("[TTS] Resumed", flush=True)


def is_paused() -> bool:
    """Return True if playback is currently paused."""
    return _paused


# ══════════════════════════════════════════════════════════════════════════════
#  QUICK TEST  (run directly: python pipertts.py)
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("  Piper TTS — Enhanced Version Test")
    print("=" * 55)

    print("\n[1/3] Ensuring model is downloaded …")
    ensure_model()

    print("\n[2/3] Pre-loading voice model …")
    t0 = time.time()
    preload()
    print(f"      Loaded in {time.time() - t0:.2f}s")

    print("\n[3/3] Playing test at different speeds …")

    speak("Testing normal speed. This is one x speed.")
    wait_until_done()

    set_speed(1.5)
    speak("Testing one point five x speed. Faster now.")
    wait_until_done()

    set_speed(2.0)
    speak("Testing two x speed. Even faster.")
    wait_until_done()

    set_speed(1.0)
    print("\n" + "=" * 55)
    print("[DONE] All tests complete.")
    print("=" * 55)

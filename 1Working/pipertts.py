"""
tts.py
======
All-in-one Piper TTS — model download, engine, and quick test.

Usage
-----
  # Run the built-in test (downloads model if needed, plays audio):
  python tts.py

  # Import and use in your own code:
  from tts import speak, announce, wait_until_done, preload

Public API
----------
  speak(text)           → synthesize and play text (non-blocking, queued)
  announce(msg)         → same as speak; use for system messages
  wait_until_done()     → block until the TTS queue is empty
  preload()             → pre-load the voice model now (avoids first-call delay)

Prerequisites
-------------
  pip install piper-tts sounddevice
  (model is downloaded automatically on first run)
"""

import os
import queue
import threading
import urllib.request

# ── Configuration ─────────────────────────────────────────────────────────────
_VOICE_NAME  = "en_US-lessac-medium"
_MODELS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "piper_models")
_MODEL_PATH  = os.path.join(_MODELS_DIR, f"{_VOICE_NAME}.onnx")
_CONFIG_PATH = os.path.join(_MODELS_DIR, f"{_VOICE_NAME}.onnx.json")
_BASE_URL    = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main"
    "/en/en_US/lessac/medium"
)


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


# ── Audio synthesis + playback ────────────────────────────────────────────────
def _synthesize(text: str):
    """Return a list of AudioChunk objects for the given text."""
    return list(_get_voice().synthesize(text))


def _play(chunks) -> None:
    """Play a list of Piper AudioChunk objects through the default audio device."""
    import numpy as np
    import sounddevice as sd

    for chunk in chunks:
        audio  = chunk.audio_int16_array           # np.ndarray of int16
        sr     = chunk.sample_rate
        ch     = chunk.sample_channels

        # Reshape to (frames, channels) so sounddevice infers channel count
        audio = np.array(audio, dtype=np.int16).reshape(-1, ch)

        # Normalize to float32 [-1, 1]
        audio_f32 = audio.astype(np.float32) / 32768.0

        sd.play(audio_f32, samplerate=sr)
        sd.wait()


# ── Background worker ──────────────────────────────────────────────────────────
_tts_queue     = queue.Queue()
_STOP_SENTINEL = object()


def _worker():
    while True:
        item = _tts_queue.get()
        if item is _STOP_SENTINEL:
            break
        try:
            _play(_synthesize(item))
        except Exception as exc:
            print(f"[TTS] Playback error: {exc}", flush=True)
        finally:
            _tts_queue.task_done()


_worker_thread = threading.Thread(target=_worker, daemon=True, name="piper-tts")
_worker_thread.start()


# ── Public API ────────────────────────────────────────────────────────────────
def speak(text: str) -> None:
    """Enqueue text for TTS. Returns immediately; audio plays in background."""
    utterance = text.strip() if text and text.strip() else "No text detected."
    _tts_queue.put(utterance)


def announce(message: str) -> None:
    """Enqueue a system announcement (e.g. 'Switched to Chinese mode.')."""
    _tts_queue.put(message.strip())


def wait_until_done() -> None:
    """Block until all queued speech has finished playing."""
    _tts_queue.join()


def preload() -> None:
    """Pre-load the voice model now to avoid the first-call delay."""
    _get_voice()


# ── Quick test (run directly: python tts.py) ──────────────────────────────────
if __name__ == "__main__":
    import time

    print("=" * 55)
    print("  Piper TTS — Quick Test")
    print("=" * 55)

    print("\n[1/3] Ensuring model is downloaded …")
    ensure_model()

    print("\n[2/3] Pre-loading voice model …")
    t0 = time.time()
    preload()
    print(f"      Loaded in {time.time() - t0:.2f}s")

    print("\n[3/3] Playing test phrases …")
    phrases = [
        "L1 Project Title The title of this project is Smart Glasses for Real Time Foreign Language Reading/Translation 1.2 Proposal Statement This project proposes the design and development of smart glasses capable of capturing primed and digital text in foreign languages and translating it into a user-selected language in real time. The system will integrate a lightweight camera mounted on wearable glasses to acquire textual information from books, instruction manuals, labels, and technical documents. The captured text will be processed using Optical Character Recognition (OCR) to convert it into machine-readable format, followed by Al-based translation algorithms to generate the translated content. The translated output will then be delivered audibly through built-in speakers or earphones, enabling hands-free real-time reading and comprehension. A functional proto- type will be developed and demonstrated under laboratory conditions to validate real-time text capture, translation accuracy, and audio output performance. 1.3 Scope of Project The scope of this project includes the design, implementation, and testing of a proof-of-concept smart glasses system for real-time foreign language reading and translation. The project shall involve integration of a camera module with a wearable gasses frame, development of OCR supporting multiple languages. An audio output subsystem shall be incorporated to deliver"
    ]
    for phrase in phrases:
        speak(phrase)

    wait_until_done()
    print("\n" + "=" * 55)
    print("[DONE] All done! Import speak() into your pipeline.")
    print("=" * 55)

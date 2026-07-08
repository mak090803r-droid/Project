# Final Demo Pipeline — Walkthrough

## What Was Built

5 files in [final/](file:///C:/Users/ali/Desktop/FYDP/New/Project/1Pipeline/final), covering **two versions** of the demo pipeline:

### Shared

| File | Size | Purpose |
|------|------|---------|
| [pipertts.py](file:///C:/Users/ali/Desktop/FYDP/New/Project/1Pipeline/final/pipertts.py) | 10.8 KB | Enhanced TTS with `set_speed()`, `pause()`, `resume()` — block-based playback for responsive controls |

### Version 1: GPIO Button Mode

| File | Size | Runs On | Purpose |
|------|------|---------|---------|
| [pipeline_nllb1.3b.py](file:///C:/Users/ali/Desktop/FYDP/New/Project/1Pipeline/final/pipeline_nllb1.3b.py) | 32.8 KB | PC | TCP server, drives full flow via commands to Pi |
| [piweb.py](file:///C:/Users/ali/Desktop/FYDP/New/Project/1Pipeline/final/piweb.py) | 23 KB | Pi | GPIO buttons, camera, audio tones, TCP client |

### Version 2: CLI / Keyboard Mode

| File | Size | Runs On | Purpose |
|------|------|---------|---------|
| [pipeline_cli.py](file:///C:/Users/ali/Desktop/FYDP/New/Project/1Pipeline/final/pipeline_cli.py) | 28 KB | PC | Live preview, S/Q keyboard, CLI prompts |
| [piweb_cli.py](file:///C:/Users/ali/Desktop/FYDP/New/Project/1Pipeline/final/piweb_cli.py) | 10.7 KB | Pi (or any PC) | Continuous stream + S/Q keyboard |

---

## Button Version Flow

```
PC: python pipeline_nllb1.3b.py          Pi: python piweb.py
┌──────────────────────────┐             ┌──────────────────────────┐
│ 1. TTS "Welcome to FYDP"│             │ Camera + GPIO init       │
│ 2. Load OCR, NLLB, UVDoc│             │ Connect to PC:9999       │
│ 3. TTS "Models loaded"  │◄───TCP──────│                          │
│ 4. TTS "Choose language" │             │                          │
│ 5. Send choose_language  │────────────►│ Wait for BTN1 pattern    │
│                          │◄────────────│ single=ZH, double=FR,    │
│ 6. TTS "Choose book mode"│             │ triple=ES                │
│ 7. Send choose_book_mode │────────────►│ single=ON, double=OFF    │
│                          │◄────────────│                          │
│ 8. TTS "Press button"   │             │                          │
│ 9. Send wait_capture     │────────────►│ Wait for BTN1            │
│                          │◄────────────│ single → 🔊beep + frame  │
│                          │             │ hold 4s → multi-capture   │
│ 10. OCR → tone cmd      │────────────►│ 🔊 rising/descending tone│
│ 11. Translate → TTS     │             │                          │
│ 12. Send tts_started     │────────────►│ Monitor BTN1 + BTN2      │
│     Handle controls     │◄────────────│ tap=pause, 2tap=stop     │
│                          │◄────────────│ BTN2=cycle 1x/1.5x/2x   │
│ 13. Send tts_ended       │────────────►│ Return to command loop   │
│ Loop back to 9           │             │                          │
└──────────────────────────┘             └──────────────────────────┘
```

### Button Controls Summary

| Phase | Button 1 | Button 2 |
|-------|----------|----------|
| Language | 1-press=Chinese, 2-press=French, 3-press=Spanish | — |
| Book Mode | 1-press=ON, 2-press=OFF | — |
| Capture | 1-press=photo, hold 4s=multi-capture | — |
| Multi-capture | 1-press=capture page, hold 4s=done | — |
| TTS Playing | 1-tap=pause/resume, 2-tap=stop | Press=cycle speed |

### Audio Tones

| Event | Tone | Duration |
|-------|------|----------|
| Photo captured | 1000Hz beep | 100ms |
| OCR found text | 400→800Hz rising | 200ms |
| No text found | 800→400Hz descending | 300ms |

---

## CLI Version Flow

```
PC: python pipeline_cli.py              Pi: python piweb_cli.py
┌──────────────────────────┐             ┌──────────────────────────┐
│ 1. TTS "Welcome"         │             │ Camera init              │
│ 2. CLI: language? (1/2/3)│             │ Connect to PC:9999       │
│ 3. CLI: book mode? (y/n) │             │ Stream frames ──────────►│
│ 4. Load models           │             │                          │
│ 5. TTS "Models loaded"   │◄───TCP──────│                          │
│ 6. Live preview window   │             │ Keyboard: S=capture      │
│ 7. S key → capture       │             │           Q=quit         │
│    OR Pi S signal ────────│◄────────────│ S pressed → signal       │
│ 8. 🔊beep → OCR → tone  │             │                          │
│ 9. Translate → TTS       │             │                          │
│ 10. Any key = stop TTS   │             │                          │
│ Loop back to 6           │             │                          │
└──────────────────────────┘             └──────────────────────────┘
```

---

## Wire Protocol

Both versions use the same protocol over TCP port 9999:

```
[1 byte: type] [payload]

Type 0x01 (JSON):  [4 bytes: uint32 BE length] [JSON UTF-8 bytes]
Type 0x02 (FRAME): [8 bytes: uint64 BE length] [pickled JPEG buffer]
```

---

## Key Design Decisions

1. **Original files untouched** — all new files in `final/`, originals in `1Pipeline/` and `1Working Comms/` preserved
2. **pipertts.py shared** by both versions — speed control via sample rate scaling (changes pitch slightly at 1.5x/2x, acceptable for demo)
3. **GPIO fallback** — button version's `piweb.py` falls back to keyboard input if `RPi.GPIO` not available (for testing on PC)
4. **Tones on Pi** (button version) or **PC** (CLI version) — feedback plays where the user is
5. **NLLB always loaded** in button version (before language selection), loaded per-language in CLI version
6. **Multi-capture** (button version only) — hold 4s to enter, press to capture pages, hold 4s to finish → all pages OCR'd, concatenated, translated once, spoken

## What to Copy to Pi

For **button mode**: copy `piweb.py` to Pi  
For **CLI mode**: copy `piweb_cli.py` to Pi  

Pi dependencies:
```
pip install opencv-python numpy sounddevice
# For button mode only:
pip install RPi.GPIO
```

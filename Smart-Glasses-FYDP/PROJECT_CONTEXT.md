# Project Handoff Documentation for Codex
**Target Recipient:** Codex AI assistant
**Project Title:** Autonomous Wearable Assistive Vision Translation Glasses (Final Year Design Project)
**Current Stage:** Mid-Project Integration (Prototype V2/V3 working on edge-host setup)

---

## 1. Project Overview & Scope
The goal of this project is to build an assistive smart reading glass system for visually impaired users. When wearing the glasses and looking at a document page, the system:
1.  **Streams** video frames wirelessly from a glass-mounted camera (Logitech C525) to a local PC host.
2.  **Analyzes** frames in real-time to assess text quality, focus sharpness, alignment, and page boundary completeness.
3.  **Triggers capture automatically** (touchless auto-capture) when the user holds still and the frame quality is optimal.
4.  **Extracts** text from the document image locally using a deep OCR engine.
5.  **Translates** the extracted text to English in real-time.
6.  **Synthesizes** natural-sounding speech and reads the translation to the user via wireless headphones/audio output.

---

## 2. System Architecture

```
 Wearable Glasses Edge Node                          Backend PC Host Node
┌─────────────────────────┐                         ┌──────────────────────────────────────────────┐
│  Logitech C525 Camera   │                         │  pipeline_cli_box3a.py (TCP Server)          │
│          │ (1080p BGR)  │                         │  ├─ socket_receiver_thread                   │
│          ▼              │                         │  │  └─ Reads Frame (Pickled JPEG)            │
│  piweb_cli.py (Client)  │                         │  ├─ score_frame_quality (Focus, Brightness)  │
│  ├─ camera_stream_thread│                         │  ├─ background_detect_text_region            │
│  │  └─ Reads frame      │                         │  ├─ auto_capture_gate (Temporal Stability)   │
│  ├─ Compress to JPEG    │                         │  ▼                                           │
│  └─ send_frame_safe     │                         │ [TRIGGER CAPTURE]                            │
└──────────┬──────────────┘                         │  ├─ preprocess_image (CLAHE + Gamma)         │
           │                                        │  ├─ PP-OCRv6_medium (onnxruntime GPU)        │
           │ (Wireless TCP / Port 9999)             │  ├─ UVDoc (Unwarping for curved pages)       │
           ▼                                        │  ├─ NLLB-200 1.3B (CTranslate2 GPU)          │
[Pickled JPEG buf / JSON CMDs]                      │  └─ Piper TTS (Local Neural Voice Output)    │
                                                    └──────────────────────────────────────────────┘
```

### 1. Hardware Split
-   **Edge Node:** Raspberry Pi 4 / Pi Zero 2W. Instantiates the camera module, captures raw frame arrays, compresses them, and handles network queue buffers.
-   **Compute Host:** Local PC containing an NVIDIA GPU. Runs CUDA-accelerated models (ONNXRuntime OCR, CTranslate2 machine translation, and Piper neural speech synthesizers).

### 2. Networking Wire Protocol
The edge node acts as a TCP client connecting to the PC host server on port `9999`:
-   **JSON Control Command Payload (Type `0x01`):** `[1 byte: Type (0x01)] [4 bytes: Big-Endian Length (uint32)] [UTF-8 JSON string bytes]`. Utilized for commands (e.g. `{"cmd": "capture_from_pi"}`).
-   **Frame Image Payload (Type `0x02`):** `[1 byte: Type (0x02)] [8 bytes: Big-Endian Length (uint64)] [Pickled JPEG buffer bytes]`.

---

## 3. Key Technical Decisions & Gating Logic

### 1. Preprocessing Sequence Optimization
During benchmarking, we evaluated 6 distinct preprocessing variants across a 20-image real dataset captured via the Logitech C525 webcam.
*   **No Gaussian Blur (V6 Winner):** Standard pipelines use `cv2.GaussianBlur` to remove noise. However, on the C525's soft plastic lens, Gaussian blur smooths out thin ink borders, causing characters (like `e` or `a`) to merge on 11-12pt text. Commenting out this step improved OCR accuracy.
*   **No PC Upscaling:** Because frames are captured in native 1080p from the camera, scaling up to `1.5x` on the PC is redundant. It introduces bilinear scaling artifacts and adds **150ms** of computational latency.
*   **Active Pipeline Sequence (Variant V6):**
    1.  **BGR Unsharp Mask (Pi):** Soft sharpening (`sigma=1.0`, `strength=1.5`) applied before encoding to improve edge definition.
    2.  **Grayscale Conversion (PC).**
    3.  **CLAHE:** Contrast adjustments to remove shadow gradients (`clipLimit=2.0`, `tileGridSize=(8, 8)`).
    4.  **Gamma Correction:** Brightens mid-tones to isolate text from paper texture (`gamma=0.7`, equivalent to lookup table inverse gamma of `1.43`).

### 2. Auto-Capture Gating Specifications
To run autonomously without physical triggers, the system monitors frames against 5 distinct gate rules:
1.  **Sharpness Gate:** Computes the variance of the Laplacian response (`cv2.Laplacian(gray, cv2.CV_64F).var()`). This value is linearly mapped from a `[50, 300]` scale to standard `0-40` points. A frame requires a total quality score of $\ge 50$ to trigger.
2.  **Density/Content Gate:** The background text detector must find $\ge 6$ text lines (`num_lines >= 6`). This prevents triggering on a single text fragment or blank desktop area.
3.  **Framing/Boundary Gate:** Computes the outer page coordinates wrapping all text lines. If the outer bounding box touches the outer frame margin ($\le 3.5\%$ of frame width, or $\le 67$px from screen edge on 1920p), the page is marked as **"PAGE CUT OFF"** and rejected. Otherwise, it is marked as **"PAGE OK"** (Green box status).
4.  **Evenness Gate:** Checks the standard deviation of average quadrant brightness. If `evenness_std >= 30`, the capture is rejected to prevent triggers under high-contrast shadow casting.
5.  **Temporal Stability Lock:** The frame must satisfy all 4 criteria consecutively for exactly **3 frames** before capture is triggered. The detection interval is set to `0.8s` (minimum hold of `2.4` seconds total).

### 3. Model Engine Comparisons
*   **OCR Engine:** PaddleOCR PP-OCRv6 (medium) selected over Tesseract. ONNXRuntime execution on GPU executes under `0.32s` (Detection + Recognition).
*   **Translation Model:** Meta's NLLB-200 1.3B selected. Compiled to CTranslate2 format (`float16` for NVIDIA CUDA GPU, fall-back to `int8` for CPU) to translate full blocks of text under `0.25s`. It handles sentence-level batch segmentation.
*   **TTS Engine:** Piper TTS (neural local voices) used with a customized `pipertts.py` wrapper, allowing play/pause queues and dynamic sample-rate scaling to alter playback speed (`1x`, `1.5x`, `2x`).

---

## 4. Codebase Directory Map

All production-ready files are located in `1Pipeline/final/`:
-   **[`piweb_cli.py`](file:///C:/Users/ali/Desktop/FYDP/New/Project/1Pipeline/final/piweb_cli.py):** Main Raspberry Pi client script. Launches a background thread to capture webcam frames, compresses them, handles keyboard listeners for overrides (`S` to capture, `Q` to quit), and transmits to the host.
-   **[`pipeline_cli_box3a.py`](file:///C:/Users/ali/Desktop/FYDP/New/Project/1Pipeline/final/pipeline_cli_box3a.py):** Main PC server script. Hosts the TCP socket server, runs the live preview window, calculates focus/evenness/framing metrics, coordinates the 3-frame stability checks, and executes the preprocessing-OCR-Translation-TTS workflow upon trigger.
-   **[`pipertts.py`](file:///C:/Users/ali/Desktop/FYDP/New/Project/1Pipeline/final/pipertts.py):** Audio synthesis manager wrapper. Contains playback speed-up, pause, and thread-safe queue logic.
-   **[`piweb.py`](file:///C:/Users/ali/Desktop/FYDP/New/Project/1Pipeline/final/piweb.py) & [`pipeline_nllb1.3b.py`](file:///C:/Users/ali/Desktop/FYDP/New/Project/1Pipeline/final/pipeline_nllb1.3b.py):** Production files for the hardware version utilizing physical GPIO buttons on the glasses instead of a CLI.

---

## 5. Benchmarks & Testing Results

Average times measured during CUDA GPU acceleration:
*   **Total pipeline execution:** **~0.9s - 1.2s**
*   **Breakdown:**
    *   Image Preprocessing: `0.08s`
    *   OCR Detection & Recognition: `0.32s`
    *   CTranslate2 NLLB Translation: `0.23s`
    *   TTS Synthesis: `0.35s`

### 📊 Preprocessing CER Comparison Table (20 Image Dataset)

| Variant | Preprocessing Description | Avg CER (Lower is Better) | Avg Speed | Accuracy Gain |
|---|---|---|---|---|
| **V1** | Raw Image (No preprocessing) | **170.70%** | 0.532s | Baseline |
| **V2** | Grayscale + GaussianBlur(3,3) + CLAHE + Gamma + 1.0x | **167.60%** | 0.507s | +1.8% |
| **V3** | No Blur + CLAHE (Grayscale + CLAHE + Gamma + 1.0x) | **165.05%** | 0.508s | +3.3% |
| **V4** | Upscale 1.5x + No Blur | **160.85%** | 0.649s | +5.8% |
| **V5** | Unsharp Mask + Upscale 1.5x | **160.99%** | 0.644s | +5.7% |
| **V6** | **Unsharp Mask + No Blur (Unsharp 1.5 + Gray + CLAHE + Gamma + 1.0x)** | **157.63%** | **0.502s** | **+7.7% (WINNER)** |

---

## 6. Known Issues, Resolved Bugs, & Workarounds

### 1. `cv2.createCLAHE` Syntax Error (Resolved)
-   *Symptom:* The pipeline crashed during startup with: `TypeError: 'tileSize' is an invalid keyword argument for createCLAHE()`.
-   *Fix:* Changed argument parameter to `tileGridSize=(8, 8)` to match OpenCV specifications.

### 2. Slow Response Time of Stability Check (Resolved)
-   *Symptom:* Auto-capture took upwards of 5 seconds to trigger even when held completely steady.
-   *Fix:* Changed `detect_interval` from `1.5s` to `0.8s` in `pre_capture_quality_loop`, cutting minimum hold times from 4.5 seconds to 2.4 seconds without lag.

### 3. Slanted Page Focus Falloff (Resolved)
-   *Symptom:* C525 fixed-lens vignette caused text lines at the top of slanted pages to blur and fail OCR.
-   *Fix:* Removed the GaussianBlur step, added soft edge unsharp mask BGR-side on Pi, and raised sharpness scoring scale ceiling to `LapVar = 300` to reject frames experiencing focus drop.

---

## 7. Next Roadmap Steps for Codex
1.  **Pi-side Model Optimization:** Quantize NLLB-200 weights to `int4` and attempt execution directly on the Pi.
2.  **Compact Hardware Housing:** Design a custom 3D model frame enclosure for the stripped Logitech C525 lens barrel and Raspberry Pi unit.
3.  **Edge-side Grayscale Streaming:** Modify `piweb_cli.py` to stream single-channel grayscale instead of BGR over socket to check if network frame rate exceeds 20 FPS.

---

*   **Local Execution Only:** The pipeline must work with zero internet connectivity. Do not replace local models with web API endpoints (e.g. OpenAI translation, cloud TTS).

---

## 9. Current Technical Debt & Infrastructure Gaps
Codex should prioritize addressing the following structural gaps immediately:

1.  **Broken Requirements File:** The root `requirements.txt` was scaffolded inside an empty directory and misses key dependencies like `piper-tts`, `paramiko`, and various hardware-specific packages.
2.  **Broken Committed Venv:** The committed `.venv` is broken and unusable; its original Python executable path is missing, and the folder contains only basic pip/setuptools. Code assistants must ignore it and rebuild their own environment.
3.  **Missing Automated Tests:** The `tests/` directory contains only a `.gitkeep` placeholder file. No automated verification scripts or tests exist.
4.  **No Configuration Management:** There is no centralized configuration file (like `config.json` or `.env`). Crucial deployment configurations (PC Server IP address, webcam properties, NLLB language maps, sharpness thresholds, and file paths) are hardcoded directly inside the python scripts.
5.  **Missing Logs & Benchmarks:** There is no structured logging system, evaluation runner, or benchmark output tracking directory to evaluate text detection and translation speeds.
6.  **Missing Claimed Files:** Diagnostic files `agent_automated_testbench.py` and `deep_image_diagnostics.py` exist only in system sandbox directories (like `.gemini/brain/`) and are missing from the main codebase source trees.
7.  **No Version Control Rules:** The codebase lacks a `.gitignore`. Compiled `.pyc` files and system temp files are being tracked, while massive local weights models and duplicate scaffolding structures remain untracked.
8.  **Incomplete Local Voice Profiles:** Only an English Piper voice profile is present locally. 
9.  **Limited Language Support:** The pipeline's target translation languages only support French, Chinese, Spanish, and English. No support is compiled or configured for Arabic, Urdu, or German.
10. **No Automatic Source Detection:** The source translation language must be hardcoded or manually selected via console/GPIO options. The pipeline cannot auto-detect what language the glasses are looking at.


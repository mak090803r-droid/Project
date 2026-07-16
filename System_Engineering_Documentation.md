# FINAL YEAR DESIGN PROJECT: SYSTEM ENGINEERING DOCUMENTATION
**System Classification:** Headless Wearable Assistive Vision System (720p C525 Input Vector)
**Host Compute Node:** Backend PC Server (ONNX Optimized Execution Runtime)

---

## 1. COMPUTE ARCHITECTURE & NETWORK LOGISTICS

The system implements a client-server topology designed to split the computational workload of wearable OCR and translation between a low-power edge node (Raspberry Pi client) and a high-performance host node (PC Server) over a local network.

### 1. Edge-to-Host Data Streaming Flow
The physical flow of data is managed over a raw TCP socket connection on port `9999`:
- **Sender (Raspberry Pi Client):** Instantiated in `piweb_cli.py`. Frames are captured from the Logitech C525 camera at a native resolution of `1920x1080` (1080p) at `15 FPS`. 
- **Transmission Format:** Frames are compressed using OpenCV's JPEG compression with a quality setting of `95%` (`cv2.IMWRITE_JPEG_QUALITY = 95`) to preserve high-frequency text boundaries while minimizing raw byte overhead.
- **Wire Protocol Format:** Data transmission over the TCP socket uses a lightweight type-length-value (TLV) framing protocol:
  - **JSON Control Payload (Type 0x01):** `[1 byte: Type (0x01)] [4 bytes: Big-Endian Length (uint32)] [UTF-8 Encoded JSON bytes]`
  - **Image Frame Payload (Type 0x02):** `[1 byte: Type (0x02)] [8 bytes: Big-Endian Length (uint64)] [Pickled JPEG buffer bytes]`

### 2. Edge Preprocessing & Grayscale Optimization
During active streaming, preview frames are downscaled to `640x360` (360p) on the Pi side. The system architecture supports converting frames to Grayscale at the edge before socket transmission. 
- **Bandwidth Reduction:** Converting a standard three-channel color buffer (BGR) to single-channel Grayscale strips **66.6%** of the raw network transmission overhead. Instead of sending 3 bytes per pixel, only 1 byte is sent.
- **Noise Elimination:** Edge-side grayscale conversion collapses chromatic aberration and color-channel noise profiles introduced by the C525 sensor, preventing color artifacts from being magnified by the JPEG compression step.

---

## 2. AUTONOMOUS CAPTURE ALGORITHMIC SPECIFICATION

To achieve touchless, hands-free operation, the PC host runs an autonomous quality control gating system. A frame is only passed to the deep OCR/Translation layers once it satisfies a series of mathematical and spatial criteria.

### 1. Sharpness/Autofocus Gate (Focus Check)
The sharpness of each incoming frame is measured using the Laplacian Variance method:
- **Calculation:** The Grayscale image is convolved with the Laplacian kernel ($\mathbf{L}$) using OpenCV's `cv2.Laplacian(gray, cv2.CV_64F)`. The variance of the resulting response map is computed:
  $$\text{Var} = \sigma^2 = \frac{1}{N}\sum_{i=1}^N (L_i - \mu)^2$$
- **Score Mapping:** The raw variance (`laplacian_var`) is mapped to a standardized `0-40` point scale inside `score_frame_quality`:
  - `laplacian_var < 50` $\rightarrow$ `0` points.
  - `laplacian_var >= 300` $\rightarrow$ `40` points (maximum focus score).
  - Intermediary values are mapped linearly:
    $$\text{sharpness\_pts} = \left\lfloor\frac{\text{laplacian\_var} - 50}{250} \times 40\right\rfloor$$
- **Gate Limit:** The total frame quality score (comprising Sharpness, Brightness, and Evenness) must exceed `QUALITY_THRESHOLD = 50`.

### 2. Density/Content Gate (Full Page Check)
To prevent the camera from triggering on random background objects, desk clutter, or blank margins, the system enforces a density constraint:
- **Rule:** A frame is only eligible for capture if the lightweight background detection thread finds at least **6 individual text bounding boxes** (`num_lines >= 6`).

### 3. Framing/Boundary Gate (Page Alignment Check)
The system tracks the outer page dimensions using a bounding box that wraps all detected text lines:
- **Logic:** The minimum and maximum X and Y coordinates of all active text lines are computed:
  $$x_{\min} = \min(x_{1}, x_{2}, \dots, x_{n}), \quad y_{\min} = \min(y_{1}, y_{2}, \dots, y_{n})$$
  $$x_{\max} = \max(x_{1}, x_{2}, \dots, x_{n}), \quad y_{\max} = \max(y_{1}, y_{2}, \dots, y_{n})$$
- **Boundary Margin:** An adaptive edge margin is calculated:
  $$\text{edge\_margin} = 0.035 \times W_{\text{frame}}$$
  For a $1920\times1080$ frame, the margin is $\approx 67$ pixels from each screen edge.
- **State Processing:** 
  - If $x_{\min} < \text{edge\_margin}$, $y_{\min} < \text{edge\_margin}$, $x_{\max} > W_{\text{frame}} - \text{edge\_margin}$, or $y_{\max} > H_{\text{frame}} - \text{edge\_margin}$, the page is flagged as **"PAGE CUT OFF"** (Red Status) and rejected.
  - If the outer box stays strictly within the inner frame limits, the page is flagged as **"PAGE OK"** (Green Status) and cleared for capture.

### 4. Temporal Stability Lock (Hold Check)
A multi-frame hysteresis loop is implemented in `pre_capture_quality_loop` to prevent trigger errors during head movements:
- **Hysteresis Loop:** A frame must pass all checks (Focus Score $\ge 50$, Content lines $\ge 6$, Green Alignment Status, and Quad Brightness Evenness $\text{QuadStd} < 30$) for exactly **3 consecutive frames** (`required_consecutive = 3`).
- **Reset Hysteresis:** If any frame in the sequence fails to satisfy any check, the stability counter immediately resets to `0`.
- **Interval:** The background detection checks are run every `0.8` seconds, meaning the user must hold the camera steady for a minimum of $3 \times 0.8\text{s} = 2.4\text{s}$ to trigger a capture.

---

## 3. IMAGE PREPROCESSING PIPELINE

Once a capture is triggered, the frame undergoes a cascading sequence of spatial filters designed to maximize character legibility for the OCR engine.

```
Raw BGR Image ──► Soft Unsharp Mask ──► Grayscale ──► CLAHE ──► Gamma Correction ──► OCR Input
```

### 1. Soft Unsharp Mask
Applied directly to the raw BGR frame prior to compression to compensate for lens focus falloff:
- **Formula:** 
  $$\text{Sharpened} = \text{Image} \times (1.0 + \text{strength}) - \text{Blurred} \times \text{strength}$$
- **Parameters:** `sigma = 1.0`, `strength = 1.5`.

### 2. Denoising & Blur Mitigation
- **Metric Configuration:** Standard `GaussianBlur(3,3)` is disabled on the C525 stream. Blur mitigation is achieved by running at native 1080p, preserving fine line boundaries of small character glyphs.

### 3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
Balances local contrast blocks to remove shadow gradients:
- **Parameters:** `clipLimit = 2.0`, `tileGridSize = (8, 8)`.

### 4. Gamma Correction
Corrects uneven exposure and mid-tone roll-off to brighten the paper page and darken ink strokes:
- **Parameters:** `gamma = 0.7` (maps to an inverse gamma of `1.43` in the lookup table).
- **LUT Equation:** 
  $$\text{LUT}[i] = \left(\frac{i}{255.0}\right)^{1.43} \times 255$$

---

## 4. CHRONOLOGICAL ENGINEERING JOURNAL & EXPERIMENTAL LOG

### Entry #1 (2026-07-10) — Webcam Shift, Slanted Focus falloff, & Stability Fixes
- **Webcam Transition:** The project migrated from the autofocus-equipped Logitech C920 camera to the fixed/plastic lens Logitech C525 camera. 
- **Focus Issues on Slanted Pages:** Initial tests on slanted A4 pages revealed significant focus falloff on the top 12 lines of text. Under standard configurations, the character error rate (CER) peaked as the letters blurred into paper textures.
- **Denoising Bottleneck:** Analyzing the preprocessing variants proved that the default `GaussianBlur(3,3)` was smoothing out high-frequency character details. Commenting out the blur step and relying on raw 1080p pixel boundaries (Variant V6) reduced the average Character Error Rate (CER) to **157.63%**, delivering a **7.7%** absolute accuracy improvement over the raw baseline.
- **Hysteresis Resolution:** Premature snapshot triggers caused by focus hunting during camera motion were solved by:
  1. Raising the sharpness mapping ceiling to `LapVar = 300` to prevent blurry frames from scoring maximum focus points.
  2. Introducing the 3-frame stability hold hysteresis loop.
  3. Adding a hard limit on quadrant lighting variation (`evenness_std < 30`).

---

## 5. CHRONOLOGICAL CODEBASE INVENTORY

All project-specific Python files inside the `New/Project` directory structure, sorted chronologically by creation date:

### 📅 July 02, 2026
- `Project\1Pipeline\1stpipeline.py` (Pipeline, 07:13:26, 14455 bytes)
- `Project\1Pipeline\pipeline2_nllb1.3b.py` (Pipeline, 07:13:26, 21911 bytes)
- `Project\1Pipeline\pipeline_nllb1.3b.py` (Pipeline, 07:13:26, 18046 bytes)
- `Project\1Pipeline\pipeline_nllb3.3b.py` (Pipeline, 07:13:26, 17353 bytes)
- `Project\1Working\3.3b nllb.py` (Translation, 07:13:26, 9371 bytes)
- `Project\1Working\argos gpu.py` (Script, 07:13:26, 5140 bytes)
- `Project\1Working\argos.py` (Script, 07:13:26, 5047 bytes)
- `Project\1Working\joined 1.3 nllb.py` (Translation, 07:13:26, 10290 bytes)
- `Project\1Working\llama 3_8b instruct.py` (Translation, 07:13:26, 17695 bytes)
- `Project\1Working\marianmt ch.py` (Translation, 07:13:26, 3324 bytes)
- `Project\1Working\marianmt fr.py` (Translation, 07:13:26, 3318 bytes)
- `Project\1Working\ocrv6 basic working.py` (OCR/CV, 07:13:26, 2105 bytes)
- `Project\1Working\opencv paddle.py` (OCR/CV, 07:13:26, 4631 bytes)
- `Project\1Working\paddle for curved pages.py` (OCR/CV, 07:13:26, 5521 bytes)
- `Project\1Working\paddleocr for curved pages.py` (OCR/CV, 07:13:26, 5521 bytes)
- `Project\1Working\pipeline.py` (Pipeline, 07:13:26, 13724 bytes)
- `Project\1Working\pipertts.py` (Network, 07:13:26, 8052 bytes)
- `Project\1Working\simple translation 1.3b nllb.py` (Translation, 07:13:26, 10171 bytes)
- `Project\1Working Comms\# stream_receiver.py` (Network, 07:13:26, 8796 bytes)
- `Project\1Working Comms\comms check rx.py` (Network, 07:13:26, 5131 bytes)
- `Project\1Working Comms\pi send.py` (Network, 07:13:26, 3348 bytes)
- `Project\1Working Comms\piweb.py` (Network, 07:13:26, 8206 bytes)
- `Project\1Working Comms\recieve coms laptop.py` (Network, 07:13:26, 2035 bytes)
- `Project\1Pipeline\pipertts.py` (Pipeline, 16:28:35, 8963 bytes)

### 📅 July 03, 2026
- `Project\test.py` (Script, 07:12:14, 63 bytes)

### 📅 July 06, 2026
- `Project\1Pipeline\final\pipertts.py` (Pipeline, 06:11:00, 11762 bytes)
- `Project\1Pipeline\final\pipeline_nllb1.3b.py` (Pipeline, 06:13:11, 32789 bytes)
- `Project\1Pipeline\final\piweb.py` (Pipeline, 06:14:39, 23027 bytes)
- `Project\1Pipeline\final\piweb_cli.py` (Pipeline, 11:16:56, 10348 bytes)
- `Project\1Pipeline\final\pipeline_cli_nocam.py` (Pipeline, 11:44:23, 22997 bytes)
- `Project\1Pipeline\final\piweb_cli_nocam.py` (Pipeline, 11:47:38, 4780 bytes)
- `Project\1Pipeline\final\upload_remote.py` (Pipeline, 14:39:48, 1153 bytes)
- `Project\1Pipeline\final\working\pipeline_cli.py` (Pipeline, 21:55:30, 30941 bytes)
- `Project\1Pipeline\final\working\piweb_cli.py` (Pipeline, 21:55:30, 10348 bytes)

### 📅 July 07, 2026
- `Project\1Pipeline\final\pipeline_cli_png.py` (Pipeline, 14:32:01, 30625 bytes)
- `Project\1Pipeline\final\pipeline_cli.py` (Pipeline, 14:35:50, 30941 bytes)
- `Project\1Pipeline\final\working\pipeline_cli_png.py` (Pipeline, 14:47:17, 30625 bytes)
- `Project\1Pipeline\final\testformat.py` (Pipeline, 14:53:35, 5576 bytes)
- `Project\1Pipeline\final\working\Ahead\pipeline_cli.py` (Pipeline, 15:15:03, 30941 bytes)
- `Project\1Pipeline\final\pipeline_cli_box.py` (Pipeline, 15:22:21, 43482 bytes)
- `Project\1Pipeline\final\pipeline_cli_box2.py` (Pipeline, 21:28:35, 48484 bytes)
- `Project\1Pipeline\final\working\pipeline_cli_box.py` (Pipeline, 21:35:39, 43503 bytes)

### 📅 July 08, 2026
- `Project\1Pipeline\final\pipeline_cli_box3.py` (Pipeline, 22:17:49, 47225 bytes)

### 📅 July 09, 2026
- `Project\1Pipeline\final\piweb_cli1.py` (Pipeline, 12:53:17, 10698 bytes)
- `Project\1Pipeline\final\pipeline_cli_box3a.py` (Pipeline, 15:01:25, 48114 bytes)
- `Project\1Pipeline\final\working\pipeline_cli_box3a.py` (Pipeline, 21:21:17, 48114 bytes)
- `Project\1Pipeline\final\working\Ahead\Further Box AutoCapture\pipeline_cli_box3a.py` (Pipeline, 21:21:33, 48114 bytes)
- `Project\1Pipeline\final\working\Ahead\Further Box AutoCapture\piweb_cli.py` (Pipeline, 21:21:33, 10348 bytes)

---
<!-- SYSTEM_ENGINEERING_DOCUMENTATION_APPEND_MARKER -->

# AGENTS.md — Developer Rules & Project Context

Welcome to the Final Year Design Project codebase. This file serves as permanent workspace context for AI coding assistants working on this system.

---

## 1. Project Architecture

This is a Headless Wearable Assistive Vision Translation system split into an Edge-Host client-server model over local TCP:
*   **Edge Client (`piweb_cli.py`):** Captures frames from a Logitech C525 camera (1080p BGR @ 15fps), compresses them to JPEG (95% quality), and streams them over TCP.
*   **Host Server (`pipeline_cli_box3a.py`):** Runs real-time preview, scoring gates (focus, density, boundary alignment, evenness), coordinates auto-capture triggers, and runs the main preprocessing, OCR, translation, and TTS pipeline.

---

## 2. Coding Conventions & Constraints

1.  **Strict Local Execution:** All translation, OCR, and TTS models must execute locally. Do not replace them with cloud APIs (e.g. Google Cloud, OpenAI, Azure).
2.  **Code preservation:** Keep code files in `1Pipeline/` and `1Working/` untouched. Always develop and edit inside `1Pipeline/final/`.
3.  **Hysteresis Handoffs:** Auto-capture relies on a 3-frame consecutive temporal stability check (`required_consecutive = 3`) to prevent triggers during autofocus hunting or motion. Always preserve this gating logic.
4.  **No Gaussian Blur:** Denoising with Gaussian blur on the C525 soft plastic lens degrades OCR accuracy for small 11-12pt text. Preprocessing must remain unblurred (Variant V6 configuration).

---

## 3. How to Run the Application

### Required Python Environment
Use the dedicated virtualenv located at:
`c:\Users\ali\Desktop\FYDP\New\Project\1Pipeline\.venv` or the global system `fydp` venv:
`c:\Users\ali\Desktop\FYDP\fydp\Scripts\python.exe`

### Running the System
1.  **Start PC Host (Server):**
    ```powershell
    c:\Users\ali\Desktop\FYDP\fydp\Scripts\python.exe pipeline_cli_box3a.py
    ```
2.  **Start Raspberry Pi (Client):**
    ```bash
    python piweb_cli.py
    ```

---

## 4. Commands for Testing

*   **Syntax & Compile Check:**
    ```powershell
    c:\Users\ali\Desktop\FYDP\fydp\Scripts\python.exe -m py_compile pipeline_cli_box3a.py
    ```
*   **Run Diagnostics:**
    ```powershell
    c:\Users\ali\Desktop\FYDP\fydp\Scripts\python.exe deep_image_diagnostics.py
    ```

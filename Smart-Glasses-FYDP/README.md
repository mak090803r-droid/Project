# Smart Glasses FYDP (Final Year Design Project)

This repository contains the wearable client and host PC pipeline scripts for the Headless Wearable Assistive Vision System.

## Project Structure

```
Smart-Glasses-FYDP/
├── AGENTS.md            # Active rules & execution environment guides for AI assistants
├── PROJECT_CONTEXT.md   # Handoff context & deep diagnostics for Codex
├── README.md            # This file
├── requirements.txt     # Global python package requirements
├── src/                 # Codebase source directory (placeholders)
└── tests/               # Codebase tests directory (placeholders)
```

## Setup & Quick Start

### Host PC Node Installation
1. Setup virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
2. Install Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the pipeline host server:
   ```bash
   python src/pipeline_cli_box3a.py
   ```

### Edge Raspberry Pi Client Installation
1. Start camera client (streams video at 1080p, quality 95):
   ```bash
   python src/piweb_cli.py
   ```

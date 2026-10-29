"""Where private data and generated output live, relative to the working directory."""
from pathlib import Path

DATA_DIR = Path("data")      # holdings, history, cash flows, overrides (gitignored)
OUTPUT_DIR = Path("output")  # agent.log, preview.html (gitignored)

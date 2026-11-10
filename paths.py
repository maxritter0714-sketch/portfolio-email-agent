"""Where private data and generated output live, relative to the working directory.

Both can be redirected with environment variables, e.g. to render the
README screenshots from the synthetic portfolio in docs/demo/.
"""
import os
from pathlib import Path

# holdings, history, cash flows, overrides (gitignored)
DATA_DIR = Path(os.environ.get("PORTFOLIO_DATA_DIR", "data"))
# agent.log, preview.html, dashboard.html (gitignored)
OUTPUT_DIR = Path(os.environ.get("PORTFOLIO_OUTPUT_DIR", "output"))

"""Root entrypoint for Streamlit Community Cloud (forwards to research_ai/app.py)."""
import os
import runpy
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
APP_DIR = ROOT_DIR / "research_ai"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

os.chdir(str(APP_DIR))
runpy.run_path(str(APP_DIR / "app.py"), run_name="__main__")

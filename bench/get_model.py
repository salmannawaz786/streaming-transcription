"""Fetch the Vosk model used by the live layer.

Whisper weights download themselves on first use; Vosk's do not. ~40MB.

    .venv/Scripts/python.exe -m bench.get_model
"""

from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
DEST = Path(__file__).resolve().parents[1] / "models"


def main() -> int:
    DEST.mkdir(exist_ok=True)
    if any(p.is_dir() and (p / "am").is_dir() for p in DEST.iterdir()):
        print(f"model already present in {DEST}")
        return 0

    print(f"downloading {URL}")
    with urllib.request.urlopen(URL) as response:
        blob = response.read()
    print(f"  {len(blob) / 1e6:.0f}MB, extracting to {DEST}")
    zipfile.ZipFile(io.BytesIO(blob)).extractall(DEST)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())

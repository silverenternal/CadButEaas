#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.vlm.v9_raster_pipeline import main

if __name__ == "__main__":
    sys.argv.insert(1, "update-docs")
    main()

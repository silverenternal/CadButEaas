#!/usr/bin/env python3
import sys
from scripts.vlm.v10_raster_pipeline import main

if __name__ == "__main__":
    sys.argv.insert(1, "train-muranet")
    main()

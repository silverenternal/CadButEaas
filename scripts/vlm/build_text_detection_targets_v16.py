#!/usr/bin/env python3
import sys
from scripts.vlm.image_only_structured_v16_pipeline import main

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("build-targets")
    main()

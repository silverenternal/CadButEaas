#!/usr/bin/env python3
import sys
from scripts.vlm.image_only_v14_pipeline import main

if __name__ == "__main__":
    sys.argv.insert(1, "relations")
    main()

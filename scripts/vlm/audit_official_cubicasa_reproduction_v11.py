#!/usr/bin/env python3
from scripts.vlm.v11_frontier_pipeline import official_baseline, parser


if __name__ == "__main__":
    args = parser().parse_args(["official-baseline"])
    official_baseline(args)

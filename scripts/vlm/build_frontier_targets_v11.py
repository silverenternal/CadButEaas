#!/usr/bin/env python3
from scripts.vlm.v11_frontier_pipeline import build_targets, parser


if __name__ == "__main__":
    args = parser().parse_args(["build-targets"])
    build_targets(args)

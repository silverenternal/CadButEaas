#!/usr/bin/env python3
from scripts.vlm.v11_frontier_pipeline import polygon_sequence_branch, parser


if __name__ == "__main__":
    args = parser().parse_args(["polygon-sequence"])
    polygon_sequence_branch(args)

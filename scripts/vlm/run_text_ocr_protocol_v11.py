#!/usr/bin/env python3
from scripts.vlm.v11_frontier_pipeline import small_object_text_detector, parser


if __name__ == "__main__":
    args = parser().parse_args(["small-object-text"])
    small_object_text_detector(args)

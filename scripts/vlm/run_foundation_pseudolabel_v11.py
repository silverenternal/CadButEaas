#!/usr/bin/env python3
from scripts.vlm.v11_frontier_pipeline import foundation_pseudolabel, parser


if __name__ == "__main__":
    args = parser().parse_args(["foundation-pseudolabel"])
    foundation_pseudolabel(args)

#!/usr/bin/env python3
from scripts.vlm.v11_frontier_pipeline import render_review_pack, parser


if __name__ == "__main__":
    args = parser().parse_args(["render-review"])
    render_review_pack(args)

#!/usr/bin/env python3
from scripts.vlm.v11_frontier_pipeline import audit_raw_text, parser


if __name__ == "__main__":
    args = parser().parse_args(["audit-raw-text"])
    audit_raw_text(args)

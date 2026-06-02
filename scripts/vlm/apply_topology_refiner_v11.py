#!/usr/bin/env python3
from scripts.vlm.v11_frontier_pipeline import topology_refiner, parser


if __name__ == "__main__":
    args = parser().parse_args(["topology-refiner"])
    topology_refiner(args)

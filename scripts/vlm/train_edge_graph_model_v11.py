#!/usr/bin/env python3
from scripts.vlm.v11_frontier_pipeline import evaluate_edge_graph, parser


if __name__ == "__main__":
    args = parser().parse_args(["edge-graph"])
    evaluate_edge_graph(args)

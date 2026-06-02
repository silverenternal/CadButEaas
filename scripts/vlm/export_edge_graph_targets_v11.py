#!/usr/bin/env python3
from scripts.vlm.v11_frontier_pipeline import export_edge_graph_targets, parser


if __name__ == "__main__":
    args = parser().parse_args(["edge-graph"])
    export_edge_graph_targets(args)

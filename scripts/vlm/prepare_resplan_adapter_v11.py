#!/usr/bin/env python3
from scripts.vlm.v11_frontier_pipeline import vector_graph_dataset_audit, parser


if __name__ == "__main__":
    args = parser().parse_args(["vector-datasets"])
    vector_graph_dataset_audit(args)

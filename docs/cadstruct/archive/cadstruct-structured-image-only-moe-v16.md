# CadStruct Structured Image-only MoE v16

v16 replaces the v15 mask/blob proposal front-end with structured raster-only experts:

- boundary graph expert for wall/opening/window line proposals
- room polygon expert for room instances
- symbol detector with FloorPlanCAD adapter
- text detector branch
- topology relation refiner

Current locked result:

```json
{
  "task": "IMG-MOE-V16-P1-010",
  "source_integrity_gate": {
    "contract_version": "image_only_moe_contract_v1",
    "checked_rows": 64,
    "passed": true,
    "violations": 0,
    "violation_counts": {},
    "sample_violations": []
  },
  "proposal_metrics": {
    "wall": {
      "tp": 133,
      "predicted": 625,
      "gold": 14180,
      "precision": 0.2128,
      "recall": 0.009379,
      "f1": 0.017967,
      "false_positive_examples": [
        {
          "prediction": {
            "id": "locked_13688_0_wall_edge_0",
            "class": "wall",
            "semantic_type": "wall",
            "family": "boundary",
            "p1": [
              392,
              129
            ],
            "p2": [
              392,
              177
            ],
            "bbox": [
              370,
              129,
              415,
              177
            ],
            "confidence": 0.5871999999999999,
            "proposal_source": "raster_boundary_graph_expert_v16"
          },
          "best_iou": 0.162716
        },
        {
          "prediction": {
            "id": "locked_13688_0_wall_edge_3",
            "class": "wall",
            "semantic_type": "wall",
            "family": "boundary",
            "p1": [
              390,
              204
            ],
            "p2": [
              390,
              256
            ],
            "bbox": [
              369,
              204,
              412,
              256
            ],
            "confidence": 0.573,
            "proposal_source": "raster_boundary_graph_expert_v16"
          },
          "best_iou": 0.039356
        },
        {
          "prediction": {
            "id": "locked_13688_0_wall_edge_1",
            "class": "wall",
            "semantic_type": "wall",
            "family": "boundary",
            "p1": [
              129,
              190
            ],
            "p2": [
              155,
              190
            ],
            "bbox": [
              129,
              179,
              155,
              202
            ],
            "confidence": 0.2988,
            "proposal_source": "raster_boundary_graph_expert_v16"
          },
          "best_iou": 0.038397
        },
        {
          "prediction": {
            "id": "locked_13688_0_wall_edge_2",
            "class": "wall",
            "semantic_type": "wall",
            "family": "boundary",
            "p1": [
              248,
              189
            ],
            "p2": [
              248,
              210
            ],
            "bbox": [
              240,
              189,
              257,
              210
            ],
            "confidence": 0.2564,
            "proposal_source": "raster_boundary_graph_expert_v16"
          },
          "best_iou": 0.10084
        },
        {
          "prediction": {
            "id": "locked_13688_0_wall_edge_4",
            "class": "wall",
            "semantic_type": "wall",
            "family": "boundary",
            "p1": [
              0,
              353
            ],
            "p2": [
              0,
              368
            ],
            "bbox": [
              0,
              353,
              1,
              368
            ],
            "confidence": 0.203,
            "proposal_source": "raster_boundary_graph_expert_v16"
          },
          "best_iou": 0.0
        },
        {
          "prediction": {
            "id": "locked_12799_1_wall_edge_1",
            "class": "wall",
            "semantic_type": "wall",
            "family": "boundary",
            "p1": [
              29,
              260
            ],
            "p2": [
              505,
              260
            ],
            "bbox": [
              29,
              78,
              505,
              442
            ],
            "confidence": 0.99,
            "proposal_source": "raster_boundary_graph_expert_v16"
          },
          "best_iou": 0.1306
```

Claim boundary: the MoE inference input is a non-SVG raster floorplan image. SVG/parser labels are offline supervision and evaluation gold only.

# CadStruct image-only MoE v17

The input contract is raster-only. The architecture is still MoE:

`raster image -> recall-first candidates -> crop evidence -> existing experts -> fusion -> scene graph`

Current evaluation summary:

```json
{
  "task": "IMG-MOE-V17-P1-007",
  "source_integrity_gate": {
    "contract_version": "image_only_moe_contract_v1",
    "checked_rows": 64,
    "passed": true,
    "violations": 0,
    "violation_counts": {},
    "sample_violations": []
  },
  "candidate_metrics": {
    "boundary": {
      "matched": 2101,
      "labeled_tp": 0,
      "precision": 0.463183,
      "recall": 0.09223,
      "f1": 0.153829,
      "label_precision": 0.0,
      "label_recall": 0.0,
      "label_f1": 0.0,
      "predicted": 4536,
      "gold": 22780
    },
    "space": {
      "matched": 363,
      "labeled_tp": 0,
      "precision": 0.79085,
      "recall": 0.511989,
      "f1": 0.621575,
      "label_precision": 0.0,
      "label_recall": 0.0,
      "label_f1": 0.0,
      "predicted": 459,
      "gold": 709
    },
    "symbol": {
      "matched": 409,
      "labeled_tp": 0,
      "precision": 0.185825,
      "recall": 0.181859,
      "f1": 0.18382,
      "label_precision": 0.0,
      "label_recall": 0.0,
      "label_f1": 0.0,
      "predicted": 2201,
      "gold": 2249
    },
    "text": {
      "matched": 83,
      "labeled_tp": 0,
      "precision": 0.030292,
      "recall": 0.128882,
      "f1": 0.049054,
      "label_precision": 0.0,
      "label_recall": 0.0,
      "label_f1": 0.0,
      "predicted": 2740,
      "gold": 644
    },
    "sheet": {
      "matched": 0,
      "labeled_tp": 0,
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0,
      "label_precision": 0.0,
      "label_recall": 0.0,
      "label_f1": 0.0,
      "predicted": 0,
      "gold": 0
    }
  },
  "expert_metrics": {
    "boundary": {
      "matched": 2101,
      "labeled_tp": 1810,
      "precision": 0.463183,
      "recall": 0.09223,
      "f1": 0.153829,
      "label_precision": 0.39903,
      "label_recall": 0.079456,
      "label_f1": 0.132523,
      "predicted": 4536,
      "gold": 22780
    },
    "space": {
      "matched": 363,
      "labeled_tp": 124,
      "precision": 0.79085,
      "recall": 0.511989,
      "f1": 0.621575,
      "label_precision": 0.270153,
      "label_recall": 0.174894,
      "label_f1": 0.212329,
      "predicted": 459,
      "gold": 709
    },
    "symbol": {
      "matched": 409,
      "labeled_tp": 0,
      "precision": 0.185825,
      "recall": 0.181859,
      "f1": 0.18382,
      "label_precision": 0.0,
      "label_recall": 0.0,
      "label_f1": 0.0,
      "predicted": 2201,
      "gold": 2249
    },
    "text": {
      "matched": 83,
      "labeled_tp": 0,
      "precision": 0.030292,
      "recall": 0.128882,
      "f1": 0.049054,
      "label_precision": 0.0,
      "label_recall": 0.0,
      "label_f1": 0.0,
      "predicted": 2740,
      "gold": 644
    }
  },
  "candidate_mean_f1": 0.201656,
  "final_mean_f1": 0.25207,
  "final_graph": {
    "rows": 64,
    "nodes": 9936,
    "edges": 4322
  },
  "expert_prediction_counts": {
    "boundary": 4536,
    "space": 459,
    "symbol": 2201,
    "text": 2740
  },
  "comparison": {
    "v15_proposal_mean_f1": 0.080493,
    "v16_proposal_mean_f1": 0.038178
  },
  "adopted": false,
  "adoption_decision": {
    "adopted": false,
    "checks": {
      "source_integrity_passed": true,
      "beats_v15_baseline": true,
      "minimum_final_mean_f1": false,
      "minimum_candidate_mean_f1": false,
      "minimum_family_f1": false,
      "typed_label_quality_nonzero": false,
      "room_and_text_predictions_present": true
    },
    "failed_checks": [
      "minimum_final_mean_f1",
      "minimum_candidate_mean_f1",
      "minimum_family_f1",
      "typed_label_quality_nonzero"
    ],
    "required_family_f1": {
      "boundary": 0.2,
      "space": 0.5,
      "symbol": 0.2,
      "text": 0.2
    },
    "v15_proposal_mean_f1": 0.080493
  },
  "failure_diagnosis": {
    "primary_failure": "raster_candidate_frontend",
    "bottlenecks": {
      "boundary": "candidate_recall",
      "space": "fusion_or_relation_quality",
      "symbol": "candidate_recall",
      "text": "candidate_recall",
      "sheet": "candidate_recall"
    },
    "fusion_scale": {
      "rows": 64,
      "nodes": 9936,
      "edges": 4322,
      "nodes_per_row": 155.25,
      "edges_per_row": 67.531
    },
    "next_required_fix": "Train or integrate a real raster detector/segmenter for boundary, room polygon, symbol, and text candidates before relying on MoE expert classification."
  },
  "source_integrity": {
    "source_mode": "image_only_raster_moe",
    "svg_candidate_ids_used": false,
    "annotation_geometry_used_at_inference": false,
    "model_input": "raster_image_only"
  }
}
```

Claim boundary:
- Inference never consumes SVG/parser/expected_json geometry.
- Offline gold is used only for audit, calibration, and locked evaluation.
- The image-only front end is not the final model; it is the candidate generator for the MoE stack.
- A passed source-integrity gate is necessary but not sufficient for adoption.

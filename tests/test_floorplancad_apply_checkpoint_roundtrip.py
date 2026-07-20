import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.floorplancad_train_line_token_panoptic_moe import (
    BASE_FEATURES,
    MODEL_FEATURE_NAMES,
    PANOPTIC_QUALITY_OBJECTIVE_VERSION,
    PANOPTIC_SEGMENT_INPUT_PROTOCOL_VERSION,
    TARGET_SCHEMA_V3,
    PANOPTIC_SPARSE_ROUTER_SCHEMA_VERSION,
    checkpoint_abi_metadata,
    geometry_decoder_config,
    make_panoptic_model,
    model_state_schema_sha256,
    objective_config_hash,
    ownership_config,
    quality_objective_contract,
    sparse_router_config,
    sq_rq_config,
    sq_rq_deployment_config,
)


torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]


def test_apply_reconstructs_nondefault_checkpoint_abi_without_ground_truth(tmp_path: Path):
    geometry = geometry_decoder_config(
        hidden_dim=16, heads=4, num_queries=8, decoder_layers=1, identity_dim=32,
        num_stuff_queries=5, local_neighbors=2, coarse_grid_size=2,
        typed_stuff_slots=True, tensor_ring_rank=2, geometry_attention_tile_size=3,
    )
    sq_rq = sq_rq_config(
        enabled=True, hidden_dim=16, heads=4, num_labels=36, gradient_scale=0.0,
        query_confidence_threshold=0.31, token_membership_threshold=0.42,
    )
    router = sparse_router_config(
        enabled=True, hidden_dim=16, num_experts=4, top_k=2, temperature=0.7,
        typed_branch_routers=True, branch_num_experts=2, branch_top_k=1,
        branch_capacity_factor=1.25, branch_dropless=True,
    )
    model = make_panoptic_model(
        torch.nn, torch, len(BASE_FEATURES), 16, 1, 4, 8, 36, 1, 0.0,
        geometry_decoder_mode="geometry_v2", num_stuff_queries=5,
        geometry_local_neighbors=2, geometry_coarse_grid_size=2,
        typed_stuff_slots=True, tensor_ring_rank=2, geometry_attention_tile_size=3,
        sq_rq_enabled=True, sq_rq_gradient_scale=0.0,
        sq_rq_query_confidence_threshold=0.31, sq_rq_token_membership_threshold=0.42,
        ownership_enabled=True, learned_sparse_router=True,
        router_num_experts=4, router_top_k=2, router_temperature=0.7,
        typed_branch_routers=True, branch_num_experts=2, branch_top_k=1,
        branch_capacity_factor=1.25, branch_dropless=True,
    )
    input_protocol = {
        "version": PANOPTIC_SEGMENT_INPUT_PROTOCOL_VERSION,
        "target_schema_version": TARGET_SCHEMA_V3,
        "input_schema_version": "floorplancad_line_json_primitive_cache_v3_segments",
        "segment_features": True,
        "content_seeded_queries": False,
    }
    metadata = checkpoint_abi_metadata(
        32,
        geometry_config=geometry,
        geometry_state_schema_sha256=model_state_schema_sha256(model.state_dict()),
        sq_rq=sq_rq,
        sq_rq_deployment=sq_rq_deployment_config(
            enabled=True,
            query_confidence_threshold=sq_rq["query_confidence_threshold"],
            token_membership_threshold=sq_rq["token_membership_threshold"],
            auto_fused=False,
            auto_fuse_reason=None,
        ),
        ownership=ownership_config(hidden_dim=16, num_queries=8),
        sparse_router=router,
        input_protocol=input_protocol,
    )
    objective_config = {
        "fixture": "apply_roundtrip",
        "quality_objective_version": PANOPTIC_QUALITY_OBJECTIVE_VERSION,
        "quality_objective_config": quality_objective_contract(),
    }
    checkpoint = {
        "schema_version": PANOPTIC_SPARSE_ROUTER_SCHEMA_VERSION,
        "state_dict": model.state_dict(),
        "feature_names": MODEL_FEATURE_NAMES,
        "hidden_dim": 16,
        "layers": 1,
        "heads": 4,
        "num_queries": 8,
        "num_labels": 36,
        "query_decoder_layers": 1,
        "dropout": 0.0,
        "position_encoding_version": metadata["position_encoding_version"],
        "quality_head": metadata["quality_head"],
        "geometry_decoder_mode": "geometry_v2",
        "geometry_config": geometry,
        "sq_rq_config": sq_rq,
        "sq_rq_deployment": metadata["sq_rq_deployment"],
        "sq_rq_auto_fused": False,
        "sq_rq_auto_fuse_reason": None,
        "ownership_config": metadata["ownership_config"],
        "gradient_control_config": None,
        "sparse_router_config": router,
        "checkpoint_abi": metadata,
        "feature_schema_sha256": metadata["feature_schema_sha256"],
        "ontology_sha256": metadata["ontology_sha256"],
        "window_contract_sha256": metadata["window_contract_sha256"],
        "input_protocol": input_protocol,
        "objective_config": objective_config,
        "objective_config_hash": objective_config_hash(objective_config),
    }
    checkpoint_path = tmp_path / "model.pt"
    torch.save(checkpoint, checkpoint_path)
    cache = tmp_path / "inference.jsonl"
    cache.write_text(json.dumps({
        "record_id": "page-1::w0000",
        "original_record_id": "page-1",
        "split": "val",
        "window_index": 0,
        "target_schema_version": TARGET_SCHEMA_V3,
        "input_schema_version": input_protocol["input_schema_version"],
        "primitive_feature_names": list(MODEL_FEATURE_NAMES),
        "inference_provenance": {
            "split": "val", "source_sha256": "a" * 64,
            "feature_schema_sha256": hashlib.sha256(b"fixture").hexdigest(), "gt_free": True,
        },
        "primitive_rows": [
            {
                "primitive_id": index,
                "features": [float(index)] * len(MODEL_FEATURE_NAMES),
                "segment_features": [
                    [float(index)] * len(MODEL_FEATURE_NAMES),
                    [float(index) + 0.1] * len(MODEL_FEATURE_NAMES),
                ],
                "segment_count": 2,
            }
            for index in range(4)
        ],
    }) + "\n", encoding="utf-8")
    output = tmp_path / "predictions.jsonl"
    report = tmp_path / "apply.json"
    trace = tmp_path / "trace.json"
    subprocess.run([
        sys.executable, "experiments/floorplancad_apply_line_token_panoptic_moe.py",
        "--model", str(checkpoint_path), "--cache", str(cache), "--output", str(output),
        "--report", str(report), "--route-trace-output", str(trace), "--ownership-decoder", "page_global",
    ], cwd=ROOT, check=True, capture_output=True, text=True)

    apply_report = json.loads(report.read_text(encoding="utf-8"))
    route_trace = json.loads(trace.read_text(encoding="utf-8"))
    assert apply_report["checkpoint_abi"]["geometry_config"]["tensor_ring_rank"] == 2
    assert apply_report["checkpoint_abi"]["geometry_config"]["geometry_attention_tile_size"] == 3
    assert apply_report["checkpoint_abi"]["sq_rq_config"]["query_confidence_threshold"] == 0.31
    assert apply_report["checkpoint_abi"]["sq_rq_config"]["token_membership_threshold"] == 0.42
    assert route_trace["rows"][0]["branch_routes"]["semantic"]["capacity"] is None

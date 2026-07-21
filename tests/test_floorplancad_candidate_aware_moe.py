import json

import pytest

from experiments import floorplancad_build_expert_candidate_proposals as expert_candidates
from experiments import floorplancad_train_line_token_panoptic_moe as train


def test_candidate_aware_queries_bias_first_thing_queries():
    torch = pytest.importorskip("torch")
    model = train.make_panoptic_model(
        torch.nn,
        torch,
        feature_dim=38,
        hidden_dim=32,
        layers=1,
        heads=4,
        num_queries=8,
        query_decoder_layers=1,
        geometry_decoder_mode="geometry_v2",
        num_stuff_queries=5,
        typed_stuff_slots=True,
        candidate_aware_queries=True,
        candidate_feature_dim=20,
    )
    x = torch.zeros((1, 4, 38), dtype=torch.float32)
    x[..., 4] = torch.linspace(0.1, 0.4, 4)
    x[..., 5] = torch.linspace(0.2, 0.5, 4)
    candidates = torch.rand((1, 2, 20), dtype=torch.float32)

    outputs = model(x, candidate_features=candidates, return_quality=True)

    assert outputs[1].shape[:2] == (1, 8)
    assert model.last_candidate_query_diagnostics["enabled"] is True
    assert model.last_candidate_query_diagnostics["candidate_count"] == 2
    assert any(key.startswith("candidate_feature_proj.") for key in model.state_dict())


def test_candidate_slots_are_reserved_when_content_seeded_queries_are_enabled():
    torch = pytest.importorskip("torch")
    model = train.make_panoptic_model(
        torch.nn,
        torch,
        feature_dim=38,
        hidden_dim=32,
        layers=1,
        heads=4,
        num_queries=8,
        query_decoder_layers=1,
        geometry_decoder_mode="geometry_v2",
        num_stuff_queries=5,
        typed_stuff_slots=True,
        content_seeded_queries=True,
        candidate_aware_queries=True,
        candidate_feature_dim=20,
    )
    x = torch.zeros((1, 4, 38), dtype=torch.float32)
    x[..., 4] = torch.linspace(0.1, 0.4, 4)
    x[..., 5] = torch.linspace(0.2, 0.5, 4)
    candidates = torch.rand((1, 2, 20), dtype=torch.float32)

    model(x, candidate_features=candidates, return_quality=True)

    assert model.last_candidate_query_diagnostics["candidate_count"] == 2
    assert model.last_query_seed_diagnostics["query_start"] == 2
    assert model.last_query_seed_diagnostics["candidate_reserved_query_count"] == 2
    assert model.last_query_seed_diagnostics["seed_count"] == 1


def test_selected_primitive_indices_respect_seed_query_start():
    torch = pytest.importorskip("torch")
    diagnostics = {
        "match_constraint_enabled": True,
        "query_start": 2,
        "seed_indices": torch.tensor([[3, 1, 99]], dtype=torch.long),
    }

    selected = train.query_selected_primitive_indices(
        torch,
        diagnostics,
        batch_index=0,
        num_queries=5,
        token_count=4,
        device=torch.device("cpu"),
    )

    assert selected.tolist() == [-1, -1, 3, 1, -1]


def test_candidate_aware_modules_are_absent_when_disabled():
    torch = pytest.importorskip("torch")
    model = train.make_panoptic_model(
        torch.nn,
        torch,
        feature_dim=38,
        hidden_dim=32,
        layers=1,
        heads=4,
        num_queries=8,
        query_decoder_layers=1,
        geometry_decoder_mode="geometry_v2",
        num_stuff_queries=5,
        typed_stuff_slots=True,
        candidate_aware_queries=False,
        candidate_feature_dim=0,
    )

    assert not any(key.startswith("candidate_feature_proj.") for key in model.state_dict())


def test_failed_candidate_forward_clears_previous_forward_state():
    torch = pytest.importorskip("torch")
    model = train.make_panoptic_model(
        torch.nn,
        torch,
        feature_dim=38,
        hidden_dim=32,
        layers=1,
        heads=4,
        num_queries=8,
        query_decoder_layers=1,
        geometry_decoder_mode="geometry_v2",
        num_stuff_queries=5,
        typed_stuff_slots=True,
        candidate_aware_queries=True,
        candidate_feature_dim=20,
    )
    x = torch.zeros((1, 4, 38), dtype=torch.float32)
    candidates = torch.zeros((1, 1, 20), dtype=torch.float32)
    model(x, candidate_features=candidates, return_quality=True)

    assert model.last_candidate_query_diagnostics is not None
    assert model.last_query_admission_logits is not None

    bad_candidates = torch.zeros((1, 1, 19), dtype=torch.float32)
    with pytest.raises(ValueError, match="candidate_features must have shape"):
        model(x, candidate_features=bad_candidates, return_quality=True)

    assert model.last_candidate_query_diagnostics is None
    assert model.last_query_admission_logits is None
    assert model.last_aux_outputs == []


def test_candidate_proposal_loader_rejects_gt_and_builds_arrays(tmp_path):
    path = tmp_path / "candidates.jsonl"
    row = {
        "record_id": "r1",
        "runtime_contract": {"gt_free": True},
        "candidates": [
            {
                "candidate_features": [float(index) for index in range(20)],
                "runtime_allowed": True,
                "primitive_ids": [1, 2],
            }
        ],
    }
    import json

    path.write_text(json.dumps(row) + "\n")
    by_record, report = train.load_candidate_proposals(path, max_candidates=4, feature_dim=20)
    features, padding, token_masks = train.candidate_arrays_for_record(
        {
            "record_id": "r1",
            "primitive_rows": [
                {"primitive_id": 1},
                {"primitive_id": 3},
                {"primitive_id": 2},
            ],
        },
        by_record,
        max_candidates=4,
        feature_dim=20,
    )

    assert report["enabled"] is True
    assert features.shape == (4, 20)
    assert padding.tolist() == [False, True, True, True]
    assert token_masks.tolist()[0] == [1.0, 0.0, 1.0]


def test_candidate_loader_pads_expert_prior_label_features(tmp_path):
    path = tmp_path / "candidates.jsonl"
    row = {
        "record_id": "r1",
        "runtime_contract": {"gt_free": True},
        "candidates": [
            {
                "candidate_features": [0.0] * 20,
                "primitive_ids": [1],
                "expert_prior": {"predicted_label": 6, "score": 0.9},
            }
        ],
    }

    path.write_text(json.dumps(row) + "\n")
    by_record, report = train.load_candidate_proposals(path, max_candidates=4, feature_dim=57)
    features, padding, _token_masks = train.candidate_arrays_for_record(
        {
            "record_id": "r1",
            "primitive_rows": [{"primitive_id": 1}],
        },
        by_record,
        max_candidates=4,
        feature_dim=57,
    )

    assert report["expert_prior_label_features"] == 1
    assert features.shape == (4, 57)
    assert padding.tolist() == [False, True, True, True]
    assert features[0, 20] == pytest.approx(0.9)
    assert features[0, 21] == pytest.approx(6 / 34)
    assert features[0, 22 + 6] == pytest.approx(0.9)


@pytest.mark.parametrize(
    "candidate_features",
    [
        [0.0] * 19,
        [float("inf")] + [0.0] * 19,
    ],
)
def test_candidate_arrays_reject_invalid_candidate_feature_rows(candidate_features):
    with pytest.raises(ValueError, match="candidate_features must be finite"):
        train.candidate_arrays_for_record(
            {
                "record_id": "r1",
                "primitive_rows": [{"primitive_id": 1}],
            },
            {
                "r1": [
                    {
                        "candidate_features": candidate_features,
                        "primitive_ids": [1],
                    }
                ]
            },
            max_candidates=4,
            feature_dim=20,
        )


def test_candidate_arrays_fall_back_to_original_page_key(tmp_path):
    path = tmp_path / "candidates.jsonl"
    row = {
        "record_id": "page-a",
        "runtime_contract": {"gt_free": True},
        "candidates": [
            {
                "candidate_features": [float(index) for index in range(20)],
                "runtime_allowed": True,
                "primitive_ids": [5],
            }
        ],
    }
    import json

    path.write_text(json.dumps(row) + "\n")
    by_record, _report = train.load_candidate_proposals(path, max_candidates=4, feature_dim=20)
    _features, padding, token_masks = train.candidate_arrays_for_record(
        {
            "record_id": "page-a::window-03",
            "original_record_id": "page-a",
            "primitive_rows": [
                {"primitive_id": 2},
                {"primitive_id": 5},
            ],
        },
        by_record,
        max_candidates=4,
        feature_dim=20,
    )

    assert padding.tolist() == [False, True, True, True]
    assert token_masks.tolist()[0] == [0.0, 1.0]


def test_candidate_activation_blocks_silent_noop_candidate_branch():
    blockers = train.candidate_activation_blockers(
        candidate_aware_queries=True,
        max_candidate_queries=4,
        candidate_feature_dim=20,
        candidate_report={"enabled": True, "candidates": 0},
        val_candidate_report={"enabled": False, "candidates": 0},
        train_candidate_coverage={"records": 3, "records_with_candidates": 0},
        val_candidate_coverage={"records": 2, "records_with_candidates": 0},
    )

    assert "train_candidate_proposals_have_no_valid_candidates" in blockers
    assert "candidate_aware_requires_validation_candidate_proposals" in blockers
    assert "train_candidate_window_coverage_zero" in blockers
    assert "validation_candidate_window_coverage_zero" in blockers


def test_candidate_coverage_allowlist_uses_base_page_key(tmp_path):
    record_path = tmp_path / "records.jsonl"
    record_path.write_text(
        json.dumps(
            {
                "record_id": "page-a::window-01",
                "primitive_rows": [{"primitive_id": 7}],
            }
        )
        + "\n"
    )
    candidate_by_record = {
        "page-a": [
            {
                "candidate_features": [0.0] * 20,
                "primitive_ids": [7],
            }
        ]
    }

    coverage = train.candidate_record_coverage(
        record_path,
        limit=None,
        record_id_allowlist={"page-a"},
        candidate_by_record=candidate_by_record,
        max_candidates=4,
        feature_dim=20,
    )

    assert coverage["records"] == 1
    assert coverage["records_with_candidates"] == 1
    assert coverage["tokens_in_candidate_masks"] == 1


def test_candidate_mask_prior_raises_selected_primitive_logits():
    torch = pytest.importorskip("torch")
    model = train.make_panoptic_model(
        torch.nn,
        torch,
        feature_dim=38,
        hidden_dim=32,
        layers=1,
        heads=4,
        num_queries=8,
        query_decoder_layers=1,
        geometry_decoder_mode="geometry_v2",
        num_stuff_queries=5,
        typed_stuff_slots=True,
        candidate_aware_queries=True,
        candidate_feature_dim=20,
        candidate_mask_prior_logit=3.0,
    )
    model.eval()
    x = torch.zeros((1, 4, 38), dtype=torch.float32)
    x[..., 4] = torch.linspace(0.1, 0.4, 4)
    x[..., 5] = torch.linspace(0.2, 0.5, 4)
    candidates = torch.zeros((1, 1, 20), dtype=torch.float32)
    candidate_masks = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]])

    without_prior = model(x, candidate_features=candidates, return_quality=True)[2].detach()
    with_prior = model(
        x,
        candidate_features=candidates,
        candidate_token_masks=candidate_masks,
        return_quality=True,
    )[2].detach()

    delta = with_prior[0, 0] - without_prior[0, 0]
    assert delta[[0, 2]].mean().item() > 2.5
    assert delta[[1, 3]].abs().max().item() < 1e-5


def test_candidate_mask_prior_survives_sq_rq_query_redecode():
    torch = pytest.importorskip("torch")
    model = train.make_panoptic_model(
        torch.nn,
        torch,
        feature_dim=38,
        hidden_dim=32,
        layers=1,
        heads=4,
        num_queries=8,
        query_decoder_layers=1,
        geometry_decoder_mode="geometry_v2",
        num_stuff_queries=5,
        typed_stuff_slots=True,
        candidate_aware_queries=True,
        candidate_feature_dim=20,
        candidate_mask_prior_logit=3.0,
        sq_rq_enabled=True,
    )
    model.eval()
    with torch.no_grad():
        model.rq_sq_query_residual_gate.fill_(2.0)
    x = torch.zeros((1, 4, 38), dtype=torch.float32)
    x[..., 4] = torch.linspace(0.1, 0.4, 4)
    x[..., 5] = torch.linspace(0.2, 0.5, 4)
    candidates = torch.zeros((1, 1, 20), dtype=torch.float32)
    candidate_masks = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]])

    without_prior = model(x, candidate_features=candidates, return_quality=True)[2].detach()
    with_prior = model(
        x,
        candidate_features=candidates,
        candidate_token_masks=candidate_masks,
        return_quality=True,
    )[2].detach()

    delta = with_prior[0, 0] - without_prior[0, 0]
    selected_delta = delta[[0, 2]].mean().item()
    unselected_delta = delta[[1, 3]].abs().max().item()
    assert selected_delta > 2.5
    assert selected_delta - unselected_delta > 2.5
    assert model.last_candidate_query_diagnostics["mask_prior_policy"].endswith("persist_after_sq_rq_redecode")


def test_candidate_mask_prior_is_disabled_by_default_even_when_candidate_aware():
    torch = pytest.importorskip("torch")
    model = train.make_panoptic_model(
        torch.nn,
        torch,
        feature_dim=38,
        hidden_dim=32,
        layers=1,
        heads=4,
        num_queries=8,
        query_decoder_layers=1,
        geometry_decoder_mode="geometry_v2",
        num_stuff_queries=5,
        typed_stuff_slots=True,
        candidate_aware_queries=True,
        candidate_feature_dim=20,
    )
    model.eval()
    x = torch.zeros((1, 4, 38), dtype=torch.float32)
    x[..., 4] = torch.linspace(0.1, 0.4, 4)
    x[..., 5] = torch.linspace(0.2, 0.5, 4)
    candidates = torch.zeros((1, 1, 20), dtype=torch.float32)
    candidate_masks = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]])

    without_prior = model(x, candidate_features=candidates, return_quality=True)[2].detach()
    with_default = model(
        x,
        candidate_features=candidates,
        candidate_token_masks=candidate_masks,
        return_quality=True,
    )[2].detach()

    assert (with_default - without_prior).abs().max().item() < 1e-5


def test_candidate_descriptor_augmentation_transforms_center_like_geometry():
    torch = pytest.importorskip("torch")
    candidates = torch.zeros((1, 1, 20), dtype=torch.float32)
    candidates[..., 1:7] = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.2, 0.3])
    params = {
        "flip_x": torch.tensor([[[True]]]),
        "flip_y": torch.tensor([[[False]]]),
        "rotations": torch.tensor([[[0]]]),
        "scale": torch.ones((1, 1, 1)),
        "shift": torch.zeros((1, 1, 1, 2)),
    }

    transformed = train.augment_candidate_descriptor_features(torch, candidates, params)

    assert transformed[0, 0, 5].item() == pytest.approx(0.8)
    assert transformed[0, 0, 6].item() == pytest.approx(0.3)


def test_candidate_descriptor_augmentation_requires_area_column():
    torch = pytest.importorskip("torch")
    candidates = torch.zeros((1, 1, 7), dtype=torch.float32)
    params = {
        "flip_x": torch.tensor([[[False]]]),
        "flip_y": torch.tensor([[[False]]]),
        "rotations": torch.tensor([[[0]]]),
        "scale": torch.ones((1, 1, 1)),
        "shift": torch.zeros((1, 1, 1, 2)),
    }

    with pytest.raises(ValueError, match="area descriptor"):
        train.augment_candidate_descriptor_features(torch, candidates, params)


def _primitive_row(primitive_id: int, x1: float, y1: float, x2: float, y2: float):
    features = [0.0] * 38
    features[0:6] = [x1, y1, x2, y2, (x1 + x2) / 2.0, (y1 + y2) / 2.0]
    features[6] = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    features[8] = 0.0
    features[9] = 1.0
    features[10] = 0.0
    features[11] = 0.01
    features[13] = 1.0
    features[31] = 0.0
    features[36] = 1.0
    return {"primitive_id": primitive_id, "features": features}


def test_expert_candidate_adapter_drops_audit_fields_and_maps_window(tmp_path):
    cache = tmp_path / "cache.jsonl"
    expert = tmp_path / "expert.jsonl"
    output = tmp_path / "candidate_proposals.jsonl"
    cache.write_text(
        json.dumps(
            {
                "record_id": "page-1::w0000",
                "original_record_id": "page-1",
                "primitive_rows": [
                    _primitive_row(10, 0.0, 0.0, 0.1, 0.0),
                    _primitive_row(11, 0.2, 0.0, 0.3, 0.0),
                    _primitive_row(12, 0.4, 0.0, 0.5, 0.0),
                ],
            }
        )
        + "\n"
    )
    expert.write_text(
        json.dumps(
            {
                "record_id": "page-1",
                "pred_instances": [
                    {
                        "primitive_ids": [10, 12, 99],
                        "label": 7,
                        "score": 0.9,
                        "source_expert": "legacy_rq_expert",
                        "gt_match": {"gt_iou": 1.0, "positive": True},
                        "semantic_id": 7,
                    }
                ],
            }
        )
        + "\n"
    )

    manifest = expert_candidates.build_candidate_file(
        cache,
        expert,
        output,
        limit_records=0,
        max_candidates_per_record=4,
        min_intersection_primitives=1,
    )
    row = json.loads(output.read_text().splitlines()[0])
    item = row["candidates"][0]
    by_record, report = train.load_candidate_proposals(output, max_candidates=4, feature_dim=20)
    _features, padding, token_masks = train.candidate_arrays_for_record(
        json.loads(cache.read_text()),
        by_record,
        max_candidates=4,
        feature_dim=20,
    )

    assert manifest["status"] == "ready"
    assert manifest["expert_source_report"]["items_with_dropped_audit_fields"] == 1
    assert row["runtime_contract"]["gt_free"] is True
    assert "gt_match" not in item
    assert "semantic_id" not in item
    assert item["primitive_ids"] == [10, 12]
    assert report["candidates"] == 1
    assert padding.tolist() == [False, True, True, True]
    assert token_masks.tolist()[0] == [1.0, 0.0, 1.0]


def test_dev_preset_preserves_explicit_expert_candidate_source(monkeypatch):
    import sys
    from pathlib import Path

    monkeypatch.setattr(sys, "argv", ["train"])
    args = train.parse_args()
    args.training_preset = "dev"
    args.candidate_aware_queries = False
    args.candidate_proposals = Path("reports/vlm/floorplancad_expert_candidate_proposals/train.jsonl")
    args.val_candidate_proposals = Path("reports/vlm/floorplancad_expert_candidate_proposals/val.jsonl")
    args.candidate_feature_dim = 20
    args.max_candidate_queries = 8
    args.candidate_ablation_tag = "legacy_expert_rq_reuse"

    train.apply_training_preset(args)

    assert args.candidate_aware_queries is True
    assert args.candidate_feature_dim == 57
    assert args.max_candidate_queries >= 64
    assert str(args.candidate_proposals).endswith("train.jsonl")
    assert str(args.val_candidate_proposals).endswith("val.jsonl")
    assert args.candidate_ablation_tag == "legacy_expert_rq_reuse"

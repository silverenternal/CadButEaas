import math
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from experiments.floorplancad_train_line_token_panoptic_moe import (
    IGNORE_LABEL,
    component_assignment_cost,
    component_targets_schema_v2,
    load_panoptic_target_arrays,
    rq_sq_quality_calibration_loss,
    weighted_mask_loss,
    weighted_semantic_loss_schema_v2,
)


def fixture(*, partial=False, count=2, capacity=256):
    rows = [
        {
            "features": [0.0] * 16,
            "semantic_id": 2,
            "instance_id": 99,
            "primitive_id": 0,
            "page_instance_id": "thing:2:canonical-a",
            "mask_loss_valid": not partial,
            "inverse_exposure_weight": 1.0,
            "log1p_primitive_length": math.log1p(1.0),
            "visible_fraction": 0.5 if partial else 1.0,
        },
        {
            "features": [1.0] * 16,
            "semantic_id": 2,
            "instance_id": 7,
            "primitive_id": 1,
            "page_instance_id": "thing:2:canonical-a",
            "mask_loss_valid": not partial,
            "inverse_exposure_weight": 0.5,
            "log1p_primitive_length": math.log1p(7.0),
            "visible_fraction": 0.5 if partial else 1.0,
        },
        {
            "features": [2.0] * 16,
            "semantic_id": 32,
            "instance_id": -1,
            "primitive_id": 2,
            "page_instance_id": "stuff:32",
            "mask_loss_valid": True,
            "inverse_exposure_weight": 1.0,
            "log1p_primitive_length": math.log1p(3.0),
            "visible_fraction": 1.0,
        },
    ]
    return {
        "target_schema_version": "floorplancad_page_window_target_v2",
        "record_id": "page::w0000",
        "original_record_id": "page",
        "window_index": 0,
        "query_target_count": count,
        "query_target_capacity": capacity,
        "query_overflow": False,
        "query_overflow_component_count": 0,
        "primitive_rows": rows,
    }


@unittest.skipIf(torch is None, "torch unavailable")
class TargetSchemaV2TrainingConsumerTest(unittest.TestCase):
    def test_semantic_gradient_uses_inverse_exposure_times_log_length(self):
        arrays = load_panoptic_target_arrays(fixture(), 16, training=True, num_queries=256)
        labels = torch.from_numpy(arrays[1])
        weights = torch.from_numpy(arrays[4])
        logits = torch.zeros((3, IGNORE_LABEL + 1), requires_grad=True)
        loss = weighted_semantic_loss_schema_v2(torch, logits, labels, weights)
        loss.backward()
        gradient_norms = logits.grad.abs().sum(dim=-1)
        expected_ratio = float(weights[1] / weights[0])
        self.assertAlmostEqual(float(gradient_norms[1] / gradient_norms[0]), expected_ratio, places=5)
        self.assertGreater(expected_ratio, 1.0)

    def test_rq_length_weights_match_log1p_protocol_without_inverse_exposure(self):
        arrays = load_panoptic_target_arrays(fixture(), 16, training=True, num_queries=256)

        np.testing.assert_allclose(
            arrays[5],
            np.asarray([math.log1p(1.0), math.log1p(7.0), math.log1p(3.0)], dtype=np.float32),
        )
        self.assertNotAlmostEqual(float(arrays[4][1]), float(arrays[5][1]))

    def test_page_instance_id_controls_identity_and_partial_mask_is_excluded(self):
        arrays = load_panoptic_target_arrays(fixture(), 16, training=True, num_queries=256)
        labels = torch.from_numpy(arrays[1])
        length_weights = torch.from_numpy(arrays[5])
        valid = torch.from_numpy(arrays[6])
        target_labels, target_masks, _weights, positives, diagnostics = component_targets_schema_v2(
            torch, labels, arrays[7], valid, length_weights, 256
        )
        self.assertEqual(positives, 2)
        self.assertEqual(int((target_masks.sum(dim=-1) == 2).sum()), 1)
        self.assertEqual(diagnostics["identity_source"], "page_instance_id")
        partial = load_panoptic_target_arrays(fixture(partial=True), 16, training=True, num_queries=256)
        _labels, masks, _weights, positives, diagnostics = component_targets_schema_v2(
            torch,
            torch.from_numpy(partial[1]),
            partial[7],
            torch.from_numpy(partial[6]),
            torch.from_numpy(partial[5]),
            256,
        )
        self.assertEqual(positives, 1)
        self.assertEqual(masks.shape[0], 1)
        self.assertEqual(diagnostics["partial_mask_components_excluded"], 1)

    def test_window_visible_policy_supervises_partial_membership_without_identity_leak(self):
        partial = load_panoptic_target_arrays(fixture(partial=True), 16, training=True, num_queries=256)
        labels, valid, weights = (torch.from_numpy(partial[index]) for index in (1, 6, 5))
        target_labels, masks, _weights, positives, diagnostics = component_targets_schema_v2(
            torch,
            labels,
            partial[7],
            valid,
            weights,
            256,
            partial_component_policy="window_visible",
            partial_component_min_tokens=2,
        )
        self.assertEqual(positives, 2)
        self.assertEqual(sorted(target_labels.tolist()), [2, 32])
        self.assertEqual(int((masks.sum(dim=-1) == 2).sum()), 1)
        self.assertEqual(diagnostics["partial_mask_components_kept_window_visible"], 1)
        self.assertEqual(diagnostics["partial_mask_components_excluded"], 0)

    def test_length_weights_change_matching_mask_and_quality_losses(self):
        query_logits = torch.zeros((2, IGNORE_LABEL + 1))
        target_labels = torch.tensor([2, 3])
        mask_logits = torch.tensor([[2.0, -2.0, 0.0], [-2.0, 2.0, 0.0]])
        target_masks = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
        uniform = torch.ones(3)
        weighted = torch.tensor([1.0, 8.0, 1.0])
        uniform_cost = component_assignment_cost(torch, query_logits, mask_logits, target_labels, target_masks, primitive_weights=uniform)
        weighted_cost = component_assignment_cost(torch, query_logits, mask_logits, target_labels, target_masks, primitive_weights=weighted)
        self.assertFalse(torch.allclose(uniform_cost, weighted_cost))
        common = dict(
            labels=target_labels,
            class_weights=torch.ones(IGNORE_LABEL + 1),
            positive_weight=1.0,
            negative_weight=1.0,
            focal_gamma=0.0,
            area_ratio_loss_weight=0.0,
            area_overcoverage_weight=1.0,
            tversky_loss_weight=0.0,
            tversky_alpha=0.3,
            tversky_beta=0.7,
            positive_prob_floor_loss_weight=0.0,
            positive_prob_floor=0.0,
        )
        uniform_mask = weighted_mask_loss(torch, mask_logits, target_masks, primitive_weights=uniform, **common)
        weighted_mask = weighted_mask_loss(torch, mask_logits, target_masks, primitive_weights=weighted, **common)
        self.assertNotAlmostEqual(float(uniform_mask), float(weighted_mask), places=5)
        q_labels = torch.tensor([2, 3])
        quality_logits = torch.tensor([1.0, 1.0])
        uniform_quality = rq_sq_quality_calibration_loss(torch, quality_logits, mask_logits, q_labels, target_masks, uniform)
        weighted_quality = rq_sq_quality_calibration_loss(torch, quality_logits, mask_logits, q_labels, target_masks, weighted)
        self.assertNotAlmostEqual(float(uniform_quality), float(weighted_quality), places=5)

    def test_capacity_212_is_accepted_by_256_and_rejected_by_96(self):
        record = fixture(count=2, capacity=212)
        self.assertIsNotNone(load_panoptic_target_arrays(record, 16, training=True, num_queries=256))
        with self.assertRaisesRegex(ValueError, "capacity/count exceed runtime num_queries"):
            load_panoptic_target_arrays(record, 16, training=True, num_queries=96)

    def test_overflow_and_legacy_training_are_fail_closed(self):
        overflow = fixture(count=2, capacity=256)
        overflow["query_overflow"] = True
        overflow["query_overflow_component_count"] = 1
        with self.assertRaisesRegex(ValueError, "query_overflow=true"):
            load_panoptic_target_arrays(overflow, 16, training=True, num_queries=256)
        legacy = {"record_id": "legacy", "primitive_rows": fixture()["primitive_rows"]}
        with self.assertRaisesRegex(ValueError, "v1/fallback cache is forbidden"):
            load_panoptic_target_arrays(legacy, 16, training=True, num_queries=256)
        with self.assertRaisesRegex(ValueError, "diagnostic-only"):
            load_panoptic_target_arrays(legacy, 16, training=True, legacy_diagnostic=True, num_queries=256)
        diagnostic = load_panoptic_target_arrays(legacy, 16, training=False, legacy_diagnostic=True, num_queries=256)
        self.assertEqual(diagnostic[0].shape, (3, 16))


if __name__ == "__main__":
    unittest.main()

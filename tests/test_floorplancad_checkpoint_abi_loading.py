import unittest

try:
    import torch
except ImportError:
    torch = None

from experiments.floorplancad_train_line_token_panoptic_moe import (
    BASE_FEATURES,
    PANOPTIC_QUALITY_OBJECTIVE_VERSION,
    checkpoint_abi_metadata,
    feature_schema_sha256,
    make_panoptic_model,
    objective_config_hash,
    ontology_sha256,
    quality_objective_contract,
    validate_checkpoint_abi,
)


@unittest.skipIf(torch is None, "torch unavailable")
class CheckpointAbiLoadingTest(unittest.TestCase):
    def tiny_model(self, seed=7, position="continuous_fourier_logspace_v2"):
        torch.manual_seed(seed)
        return make_panoptic_model(
            torch.nn,
            torch,
            feature_dim=len(BASE_FEATURES),
            hidden_dim=8,
            layers=1,
            heads=2,
            num_queries=3,
            query_decoder_layers=1,
            dropout=0.0,
            position_encoding_version=position,
        )

    def production_checkpoint(self):
        model = self.tiny_model()
        metadata = checkpoint_abi_metadata(32)
        objective_config = {
            "fixture": True,
            "quality_objective_version": PANOPTIC_QUALITY_OBJECTIVE_VERSION,
            "quality_objective_config": quality_objective_contract(),
        }
        return {
            "schema_version": "floorplancad_line_token_panoptic_moe_checkpoint_v2",
            "state_dict": model.state_dict(),
            "feature_names": BASE_FEATURES,
            "position_encoding_version": metadata["position_encoding_version"],
            "quality_head": metadata["quality_head"],
            "checkpoint_abi": metadata,
            "feature_schema_sha256": feature_schema_sha256(),
            "ontology_sha256": ontology_sha256(),
            "window_contract_sha256": metadata["window_contract_sha256"],
            "objective_config": objective_config,
            "objective_config_hash": objective_config_hash(objective_config),
        }

    def test_valid_production_abi_is_accepted(self):
        report = validate_checkpoint_abi(self.production_checkpoint())
        self.assertEqual(report["status"], "validated")
        self.assertTrue(report["production_compatible"])
        self.assertTrue(report["quality_head_trained"])
        self.assertFalse(report["quality_admission_promoted"])
        self.assertEqual(report["quality_objective_version"], PANOPTIC_QUALITY_OBJECTIVE_VERSION)
        self.assertTrue(report["quality_admission_compatible"])

    def test_nondefault_quality_mask_threshold_is_cross_bound(self):
        model = self.tiny_model()
        metadata = checkpoint_abi_metadata(32, quality_mask_threshold=0.3)
        objective_config = {
            "fixture": True,
            "quality_objective_version": PANOPTIC_QUALITY_OBJECTIVE_VERSION,
            "quality_objective_config": quality_objective_contract(mask_threshold=0.3),
        }
        checkpoint = {
            "schema_version": "floorplancad_line_token_panoptic_moe_checkpoint_v2",
            "state_dict": model.state_dict(),
            "feature_names": BASE_FEATURES,
            "position_encoding_version": metadata["position_encoding_version"],
            "quality_head": metadata["quality_head"],
            "checkpoint_abi": metadata,
            "feature_schema_sha256": feature_schema_sha256(),
            "ontology_sha256": ontology_sha256(),
            "window_contract_sha256": metadata["window_contract_sha256"],
            "objective_config": objective_config,
            "objective_config_hash": objective_config_hash(objective_config),
        }
        report = validate_checkpoint_abi(checkpoint)
        self.assertEqual(report["status"], "validated")
        self.assertEqual(
            checkpoint["checkpoint_abi"]["quality_objective_config"]["mask_threshold"],
            0.3,
        )

    def test_quality_promotion_state_is_cross_bound(self):
        checkpoint = self.production_checkpoint()
        checkpoint["checkpoint_abi"]["quality_admission_promoted"] = True
        with self.assertRaisesRegex(ValueError, "promotion state is inconsistent"):
            validate_checkpoint_abi(checkpoint)

        checkpoint["quality_admission_promoted"] = True
        checkpoint["checkpoint_boundary"] = "best_selection_checkpoint"
        checkpoint["selection_gate"] = {"passed": True}
        checkpoint["selection_score"] = 0.25
        report = validate_checkpoint_abi(checkpoint)
        self.assertTrue(report["quality_admission_promoted"])

        checkpoint["checkpoint_boundary"] = "last_epoch_diagnostic_checkpoint"
        with self.assertRaisesRegex(ValueError, "gate-passed best checkpoint"):
            validate_checkpoint_abi(checkpoint)

    def test_weights_only_quality_objective_migration_is_explicit(self):
        checkpoint = self.production_checkpoint()
        legacy_contract = {
            "version": "group_balanced_decoded_hard_mask_iou_bce_logit_margin_v4",
            "target": "continuous_hard_mask_iou",
            "negative_target": 0.0,
            "mask_threshold": 0.5,
            "metric_weighting": "unweighted_primitive_set",
            "decoder_scope": "window_ownership_null_competition_premerge_when_available",
            "deployment_use": "class_probability_times_quality_probability",
        }
        metadata = checkpoint["checkpoint_abi"]
        metadata["quality_objective_version"] = legacy_contract["version"]
        metadata["quality_objective_config"] = legacy_contract
        metadata["quality_objective_config_sha256"] = objective_config_hash(legacy_contract)
        metadata.pop("quality_admission_promoted")
        checkpoint["objective_config"]["quality_objective_version"] = legacy_contract["version"]
        checkpoint["objective_config"]["quality_objective_config"] = legacy_contract
        checkpoint["objective_config_hash"] = objective_config_hash(checkpoint["objective_config"])

        with self.assertRaisesRegex(ValueError, "checkpoint ABI mismatch"):
            validate_checkpoint_abi(checkpoint)
        report = validate_checkpoint_abi(
            checkpoint,
            allow_quality_objective_mismatch=True,
        )
        self.assertTrue(report["quality_objective_migration_allowed"])
        self.assertFalse(report["quality_admission_compatible"])

    def test_pre_all_query_quality_objective_is_rejected(self):
        checkpoint = self.production_checkpoint()
        del checkpoint["checkpoint_abi"]["quality_objective_version"]
        with self.assertRaisesRegex(ValueError, "quality_objective_version"):
            validate_checkpoint_abi(checkpoint)

    def test_objective_hash_and_quality_cross_binding_fail_closed(self):
        checkpoint = self.production_checkpoint()
        checkpoint["objective_config"]["fixture"] = False
        with self.assertRaisesRegex(ValueError, "objective config hash mismatch"):
            validate_checkpoint_abi(checkpoint)

        checkpoint = self.production_checkpoint()
        checkpoint["objective_config"]["quality_objective_config"] = {
            **quality_objective_contract(),
            "mask_threshold": 0.25,
        }
        checkpoint["objective_config_hash"] = objective_config_hash(checkpoint["objective_config"])
        with self.assertRaisesRegex(ValueError, "quality objective config is not cross-bound"):
            validate_checkpoint_abi(checkpoint)

    def test_wrong_or_incomplete_abi_is_rejected(self):
        checkpoint = self.production_checkpoint()
        checkpoint["checkpoint_abi"] = dict(checkpoint["checkpoint_abi"])
        checkpoint["checkpoint_abi"]["position_encoding_version"] = "continuous_fourier_legacy_v1"
        with self.assertRaisesRegex(ValueError, "checkpoint ABI mismatch"):
            validate_checkpoint_abi(checkpoint)
        checkpoint = self.production_checkpoint()
        del checkpoint["checkpoint_abi"]["ontology_sha256"]
        with self.assertRaisesRegex(ValueError, "metadata incomplete"):
            validate_checkpoint_abi(checkpoint)

    def test_partial_quality_state_is_always_rejected(self):
        checkpoint = self.production_checkpoint()
        del checkpoint["state_dict"]["query_quality_head.bias"]
        with self.assertRaisesRegex(ValueError, "partial query_quality_head"):
            validate_checkpoint_abi(checkpoint)
        checkpoint.pop("checkpoint_abi")
        with self.assertRaisesRegex(ValueError, "partial query_quality_head"):
            validate_checkpoint_abi(checkpoint, legacy_position_compat=True)

    def test_legacy_requires_explicit_flag_and_uses_untrained_multiplier_one(self):
        state = self.tiny_model().state_dict()
        state.pop("query_quality_head.weight")
        state.pop("query_quality_head.bias")
        checkpoint = {"state_dict": state, "feature_names": BASE_FEATURES}
        with self.assertRaisesRegex(ValueError, "ABI metadata missing"):
            validate_checkpoint_abi(checkpoint)
        report = validate_checkpoint_abi(checkpoint, legacy_position_compat=True)
        self.assertEqual(report["position_encoding_version"], "continuous_fourier_legacy_v1")
        self.assertFalse(report["quality_head_trained"])
        self.assertEqual(report["quality_multiplier"], 1.0)
        self.assertFalse(report["production_compatible"])

    def test_explicit_legacy_query_and_mask_outputs_are_seed_stable(self):
        source = self.tiny_model(seed=11, position="continuous_fourier_legacy_v1")
        state = source.state_dict()
        state.pop("query_quality_head.weight")
        state.pop("query_quality_head.bias")
        outputs = []
        features = torch.randn((1, 5, len(BASE_FEATURES)), generator=torch.Generator().manual_seed(99))
        for seed in (17, 29):
            model = self.tiny_model(seed=seed, position="continuous_fourier_legacy_v1")
            incompatible = model.load_state_dict(state, strict=False)
            self.assertEqual(set(incompatible.missing_keys), {"query_quality_head.weight", "query_quality_head.bias"})
            model.eval()
            with torch.no_grad():
                _semantic, query, masks, _random_untrained_quality = model(features, return_quality=True)
            outputs.append((query, masks))
        self.assertTrue(torch.equal(outputs[0][0], outputs[1][0]))
        self.assertTrue(torch.equal(outputs[0][1], outputs[1][1]))


if __name__ == "__main__":
    unittest.main()

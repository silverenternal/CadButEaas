# Domain-Structured MoE Main Claim v1

## Main Claim

CadStruct-MoE is best framed as an auditable domain-structured MoE for typed floorplan scene-graph parsing. The main router is deterministic over typed candidate streams, not a generic sparse-token Vision-MoE router.

## Evidence

- Deterministic structured router: wrong_expert_rate=0.0, route_accuracy=1.0, candidates=134043.
- Learned fair router ablation: wrong_expert_rate=0.152302, route_accuracy=0.847698; this is worse for the main claim.
- Largest learned-router confusions: {"room_space->symbol_fixture": 12265, "wall_opening->symbol_fixture": 2581, "room_space->wall_opening": 1938, "symbol_fixture->room_space": 1201, "wall_opening->text_dimension": 874, "text_dimension->wall_opening": 495, "symbol_fixture->wall_opening": 449, "text_dimension->symbol_fixture": 367, "wall_opening->room_space": 73, "symbol_fixture->text_dimension": 68, "room_space->text_dimension": 64, "text_dimension->room_space": 40}.
- Core measured experts: wall_opening, room_space, symbol_fixture, text_dimension.
- Replay/fusion p50 latency: 2653.591 ms; peak RSS: 1098.203 MB. This excludes OCR/VLM/expert inference as stated in the resource table.
- Lie/SE(2) geometry is now supported as a core accuracy component by `reports/vlm/lie_se2_multiseed_matched_ablation_v3.json`, with transform generalization limited by v9.
- External OCR/cross-source symbol generalization remains blocked by `reports/vlm/external_human_gold_manifest_v3.json` until human gold is filled.

## Allowed Wording

CadStruct-MoE uses deterministic domain-structured routing and family-specialized experts to produce typed floorplan scene graphs with auditable route boundaries, measured expert contributions, and explicit resource accounting.

## Blocked Wording

- Do not claim generic sparse-token MoE superiority.
- Do not claim learned/top-k routing is the main mechanism.
- Do not claim cross-source/wild generalization while external human-gold status is pending.
- Do not claim Lie/SE(2) image-level transform generalization from the current v9 stress test.

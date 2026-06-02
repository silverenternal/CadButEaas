# CadStruct Paper Core Contributions v2

## Positioning

CadStruct-MoE is an auditable domain-structured MoE system for typed CAD/floorplan scene-graph parsing. The paper should emphasize typed routing, no-repair relation fusion, explicit gated Lie/SE(2) geometry, and reproducible claim boundaries.

## Main Metrics

- Node macro F1: 0.951696
- Node accuracy: 0.981566
- No-repair relation F1: 0.920938
- Relation precision/recall: 0.961937 / 0.88329
- Invalid graph rate: 0.0

## Domain-Structured MoE

The deterministic structured router is the main router. `reports/vlm/domain_structured_moe_main_router_table_v1.json` reports wrong_expert_rate=0.0, while the learned fair router ablation remains at wrong_expert_rate=0.152302. `reports/vlm/expert_contribution_matrix_main_v1.json` identifies wall_opening, room_space, symbol_fixture, and text_dimension as measured core experts; sheet_layout is a non-core extension in the current graph.

## Lie/SE(2)

`reports/vlm/lie_se2_core_claim_decision_v9.json` supports the explicit gated Lie/SE(2) residual branch as a core accuracy component. The supported claim is matched/identity performance improvement, including h512 mean smoke macro-F1 gain of 1.822pp and seed30 identity gains of 1.227pp vs ungated full-Lie and 3.319pp vs no-Lie.

The current v9 stress test does not support image-level or broad coordinate-transform generalization; keep that as a blocked claim.

## External Boundary

`reports/vlm/external_generalization_claim_decision_v3.json` confirms the external OCR and cross-source symbol packs are annotation-ready, but human-gold counts are still zero. External OCR, cross-source symbol, and WAFFLE/ResPlan-style in-the-wild generalization remain blocked until human gold is filled.

## Allowed Claim

CadStruct-MoE combines deterministic domain-structured expert routing, explicit gated Lie/SE(2) geometry, and conservative no-repair relation scoring to produce typed floorplan scene graphs with strong locked-split node/relation metrics and auditable claim boundaries.

## Blocked Claims

- Do not claim 99% guaranteed SCI2 acceptance.
- Do not claim generic sparse-token Vision-MoE superiority.
- Do not claim broad external/wild floorplan generalization.
- Do not claim Lie/SE(2) as the sole or dominant source of the full system accuracy.
- Do not claim Lie/SE(2) image-level transform generalization.
- Do not claim repair-enabled relation scores as main-table evidence.

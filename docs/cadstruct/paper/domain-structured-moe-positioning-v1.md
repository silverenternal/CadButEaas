# Domain-Structured MoE Positioning v1

CadStruct-MoE should be positioned as a domain-structured mixture-of-experts system, not as a generic sparse-token Vision-MoE. The main router uses typed floorplan candidate streams and deterministic family assignment; learned routing is reported as an ablation/future-work path.

Current paper-main E2E result: node macro F1=0.951696, relation F1=0.920938, invalid graph rate=0.0 from `reports/vlm/scene_graph_fusion_symbol_long_tail_model_no_repair_scorer_v1_eval.json`.

Routing evidence:
- Deterministic structured router: wrong_expert_rate=0.0, abstain_rate=0.0.
- Fair learned geometry/page-context router: wrong_expert_rate=0.152302, abstain_rate=0.034668; this is not strong enough for the main model.
- Top-2/top-3 routing improves oracle family coverage but increases expected expert compute by 2x/3x and remains capacity diagnostic unless downstream graph gains are shown.

Expert specialization evidence:
- RoomSpace has the largest drop-one node macro impact in the current contribution matrix.
- TextDimension and WallOpening also provide measured node-labeling contributions.
- SymbolFixture remains the long-tail bottleneck and should be framed as a target for future symbol-model strengthening rather than hidden inside the router claim.
- SheetLayout remains a non-core extension because it has no current measured real-upstream nodes.

Allowed claim: CadStruct-MoE uses auditable domain-structured routing and family-specialized experts for typed floorplan scene-graph parsing. This is different from generic Vision-MoE routing, where the novelty is sparse token dispatch rather than engineering-domain decomposition and claim-ledger reproducibility.

Blocked claims:
- Do not claim learned sparse routing is the main contribution.
- Do not claim top-k routing improves downstream metrics without a formal graph-metric adoption report.
- Do not claim SheetLayout is a measured core expert.

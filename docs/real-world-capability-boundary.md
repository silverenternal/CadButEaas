# Real-World Capability Boundary

CadStruct-MoE is currently a research system for structure-aware floor-plan recognition. It is not yet a universal production recognizer for arbitrary engineering drawings.

## Supported With Strong Evidence

| Element Family | Current Evidence | Boundary |
| --- | --- | --- |
| Wall/opening recognition | Locked mixed-source WallOpening metrics exceed 98% macro F1 and 99% accuracy in the selected protocol. | Strongest claim; still needs broader source-heldout evidence before claiming universal generalization. |
| Basic room typing with gold/proposal assistance | RoomSpace reports separate gold-polygon and proposal-assisted evaluation. | Do not claim full end-to-end room recognition without proposal recall and mask/polygon quality. |
| Constraint scene-graph fusion | Smoke expected-json evaluation reports high relation F1 and invalid graph rate of zero. | This is fusion evidence, not proof that upstream detectors solve real scenes. |
| Engineering reproducibility | Shared expert schema, memory estimator, tile audit, source registry, and hard-case mining are present. | These are reproducibility and audit claims, not accuracy claims. |

## Partially Supported

| Element Family | Gap |
| --- | --- |
| SymbolFixture | Current crop/context encoder attempts remain below paper-grade macro F1; sink/equipment, stair/column, and appliance/equipment confusions remain central. |
| TextDimension | Teacher-assisted upper bound is useful, but deployable baseline remains below target; OCR exactness and dimension relation extraction need stronger evidence. |
| Source-heldout generalization | WallOpening has valid LOSO coverage, but other experts are blocked by single locked-source coverage. |
| Zero-shot VLM comparison | Cached VLM runs are below the required 30 samples per model; treat as smoke only. |

## Not Yet Supported

- Complex MEP/HVAC/electrical symbols beyond the current SymbolFixture ontology.
- Non-standard legends where symbol meaning is only defined in a local key.
- Very low-quality scans with folds, shadows, bleed-through, or heavy compression artifacts.
- Multi-page drawing sets requiring sheet-to-sheet reference resolution.
- Multilingual dense annotations beyond the current TextDimension/OCR protocol.
- Arbitrary CAD domains such as mechanical assemblies, PCB layouts, or process diagrams.

## Claim Rules

Use strict metrics as the main paper and README result. Do not use ambiguity-adjusted or teacher-backed numbers as the headline result.

Do not claim "98% real-world drawing recognition" unless each relevant element family has locked-test or source-heldout evidence above the target.

Do not claim 14B/VLM fine-tuning is the core solution. Current evidence supports using large VLMs as teacher, weak-label assistant, OCR/layout helper, and zero-shot baseline.

The honest current claim is: CadStruct-MoE provides a structure-aware, auditable MoE route for floor-plan recognition, with strong wall/opening results and a reproducible plan for extending symbols, text, and scene graphs.

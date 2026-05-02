# Real Drawing Annotation Protocol v1

This protocol defines the minimum review contract for `internal_real_v3`
candidate pages. Records remain review-only until two reviewers complete the
fields below and conflicts are resolved.

## Scope

- Element families: wall, opening, room, symbol, text, dimension, layout, and
  scene graph relations.
- Sources: CubiCasa5K, CVC-FP, FloorPlanCAD, and internal real drawings when
  privacy review allows use.
- Splits: review candidates first; locked candidates cannot be used for
  training or threshold tuning.

## Required Page Fields

- `source`: dataset or internal source bucket.
- `license_or_privacy_status`: public license or internal privacy status.
- `scan_quality`: one of `clean_raster`, `public_benchmark_raster`,
  `color_scan_like`, `degraded_candidate`, or `unknown_requires_review`.
- `expected_element_groups`: expected families visible in the candidate.
- `review_status`: `pending_human_review`, `reviewed`, or `conflict`.

## Required Review Fields

- `reviewer_a_label` and `reviewer_b_label`: normalized element or relation
  label.
- `conflict_status`: `unreviewed`, `agreement`, `conflict`, or
  `needs_adjudication`.
- `uncertainty_tags`: comma-separated tags such as `ambiguous_symbol`,
  `low_quality_scan`, `partial_crop`, `ocr_unclear`, `missing_context`, or
  `privacy_blocked`.
- `review_decision`: `accept_locked`, `accept_train_only`, `reject`,
  `needs_redaction`, or `needs_more_context`.
- `notes`: short reviewer or adjudicator note.

## Family Guidelines

- Wall/opening: label hard walls, doors, windows, and ambiguous openings. Mark
  missing host-wall context with `missing_context`.
- Room: label room type only when the room polygon or proposal is visually
  supportable. Use `ambiguous_room_boundary` for unclear boundaries.
- Symbol: label fixture/equipment/stair/column/appliance classes when visible.
  Use `unknown_symbol` for non-standard or legend-only symbols.
- Text/dimension: record normalized visible text when OCR is possible. Mark
  empty, cropped, or illegible text with `ocr_unclear`.
- Layout: identify title blocks, legends, schedules/tables, stamps, and notes
  regions that should be isolated from geometry recognition.
- Scene graph: review relations including `bounds`, `contains`,
  `attached_to`, `adjacent_to`, `labeled_by`, `dimension_of`, and
  `callout_of`.

## Conflict Handling

If reviewer labels differ, set `conflict_status` to `conflict` and keep both
labels. An adjudicator may set `review_decision` only after writing a note that
explains the chosen label or rejection reason.

## Output Pack

The initial review pack is:

- `reports/vlm/internal_real_v3_review_pack/review_queue.jsonl`
- `reports/vlm/internal_real_v3_review_pack/review_queue.csv`
- `reports/vlm/internal_real_v3_review_pack/review.html`


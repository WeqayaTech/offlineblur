# Results by pipeline — visual comparison index

One folder per pipeline, in the order they were tried. Each has a `PIPELINE.md` brief (what ran, where, settings,
speed, accuracy). Only the briefs are versioned: the videos and the `metrics/` folders (run JSON, track dumps,
hand labels) stay on the machine that produced them. Videos use the same names everywhere, so the
same file across folders is directly comparable:

- `labels_masks_ids*.mp4` — unblurred, masks + ids + gender (two renderers; see each PIPELINE.md for the legend)
- `blur_side_by_side.mp4` — original left, blurred right (the product)

Same clip everywhere: `ali-dawah-street-interview-source.mp4` (300 frames, 1280x720, 25 fps, 12 s).

| Folder | Detector + mask | Tracker | Gender decision | GPU | Rate | Peak GPU | Women blurred | Men false-blur |
|---|---|---|---|---|---|---|---|---|
| 01 SAM 3 vs SAM 3.1, person only | SAM 3 / SAM 3.1 (`person`) | built in | — | L4 | 0.39 / 1.84 Hz | 8.4 / 15.1 GB | — | — |
| 02 SAM 3.1 tracked, gender prompts | SAM 3.1 (`woman`,`man`,`person`) | built in (Object Multiplex) | the prompt | L4 | 0.69 Hz | 16.2 GB | 0.737 | 0.168 |
| (SAM 3 tracked, gender prompts — metrics only, in 02) | SAM 3 | built in | the prompt | L4 | 0.18 Hz | 16.4 GB | 0.799 | 0.214 |
| 03 SAM 3.1 detector only | SAM 3.1 (`woman`) | none | the prompt | L4 | 3.9 Hz | 8.1 GB | 0.743 | 0.099 |
| 04 SAM 3.1 det + McByte masks | SAM 3.1 (`woman`,`man`,`child`) | McByte + SAM/Cutie | per-track argmax vote | L4 | 1.35 Hz | 8.3 GB | 0.728 | 0.026 |
| 05 SAM 3.1 det + McByte box, share ≥ 0.4 | SAM 3.1 (`woman`,`man`,`child`) | McByte box | woman share ≥ 0.4 | L4 | 2.7 Hz | 8.3 GB | 0.731 | 0.039 |
| 06 RF-DETR + McByte + CLIP (scripts) | RF-DETR-Seg 2XL | McByte box | CLIP ViT-L/14-336, K=10, ≥ 0.25 | L4 | 11.4 Hz | 3.1 GB | 0.696 | 0.029 |
| 07 same, fast in-memory | RF-DETR-Seg 2XL | McByte box | CLIP ViT-L/14-336, K=10, ≥ 0.25 | RTX PRO 6000 | **85–93 Hz** | 4.7 GB | ≈ 06 | ≈ 06 |
| 08 fast, McByte threshold sweep | RF-DETR-Seg 2XL | McByte box, 0.5→0.2 | CLIP | RTX PRO 6000 | 72–86 Hz | 4.8 GB | not scored | not scored |
| 09 fast, lost tracks kept 4 s | RF-DETR-Seg 2XL | McByte box, lost 1 s → 4 s | CLIP | L4 | 19.5–20 Hz | 4.7 GB | not scored | not scored |
| 10 fast, association tuning (4 s) | RF-DETR-Seg 2XL | McByte, buffered IoU 0.5 + 2nd round 0.3 | CLIP | L4 | 20 Hz | 4.7 GB | not scored | not scored |
| 11 fast, classifier swap | RF-DETR-Seg 2XL | McByte, as 10 | CLIP ViT-L 18.8 Hz vs **PE-Core-L** 17.4 Hz | L4 | 17.4–18.8 Hz | 4.7 / 6.1 GB | not scored | not scored |
| 12 **EdgeTAM** tracker | RF-DETR-Seg 2XL | EdgeTAM, one memory per person | PE-Core-L | L4 | 3.6 Hz | 6.6 GB | best by eye, not scored | not scored |

Rates are frames per second of processing (1 / compute per frame; 07–08 are end to end incl. decode, blur and
encode, excluding model load). "Women blurred" / "men false-blur" are on the 21 hand-labelled people (labels and
evaluations kept locally with the metrics, scored with `tracking/gender_gt_eval.py`).

Suggested comparisons:
- **Tracking quality**: `01/person_tracking_sam3_left_vs_sam31_right.mp4` vs `07/labels_masks_ids.mp4` vs
  `08/grid_2x2_*.mp4` — SAM keeps more background people and longer ids; McByte tracks ~13–18 people per frame.
- **Gender decision**: `02/labels_masks_ids.mp4` (prompt, flips per frame) vs `04/labels_masks_ids.mp4` (per-track
  vote) vs `07/labels_masks_ids.mp4` (per-track CLIP).
- **The product**: every `blur_side_by_side.mp4`, especially 02 vs 05 vs 07.
- **Tracker, McByte vs EdgeTAM**: `12/labels_mcbyte_left_edgetam_right.mp4` and `12/blur_original_mcbyte_edgetam.mp4`.
- **Classifier, CLIP vs PE-Core**: `11/labels_clip_left_pecore_right.mp4`.
- **Occlusion handling**: `09/ids_1s_left_vs_4s_right.mp4` (longer memory alone barely helps) vs
  `10/grid_2x2_*.mp4` (buffered-IoU matching halves identity breaks).

Full write-up with charts: https://claude.ai/artifact/LCjvGwqZ9ZFTy333GXRGkD (tracking/reports/pipeline_benchmarks.html).
L4 latency of every pipeline, by stage: https://claude.ai/artifact/24RvgFzPcBApVq1oGXroR8 (tracking/reports/l4_pipeline_latency.html).

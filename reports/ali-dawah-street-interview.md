# Clip 1 — ali-dawah-street-interview-source.mp4 (owner-supplied)

1280x720, 25 fps, 12 s, night-time London street, reporter interviewing a woman, ~25 pedestrians.
Pod: RunPod RTX PRO 6000 Blackwell (96 GB). Five runs, one per code iteration.

## Run history

| run | code | identities | blurred as Woman | false blurs | escapes | notes |
|---|---|---|---|---|---|---|
| 1 | v1.0 (YOLO11x-seg + BoT-SORT → DINOv2/ArcFace merge → Qwen vote) | 60 | 10 (5 uncertain) | stroller woman's id handed to a man for a few frames (tracker id switch) | red-coat woman's strip behind the reporter judged "man" (1.4 s) | 0 face links: faces too small at 720p full-frame detection |
| 2 | v1.1 (OSNet body re-id at 0.72, upscaled-crop faces, handoff) | 35 | 6 | — | 3 women merged INTO men's identities by body links (red coat, orange jacket, pink bag) → unblurred | body similarity between strangers reaches 0.83 on this clip |
| 3 | v1.2 (classify per segment → link with class agreement; body 0.85 + motion; face 0.45; duplicate attach; vote split) | 69 | 13 (6 uncertain) | reporter's own face fragment blurred 0.28 s (uncertain Woman 3:2) | 3 short gap-split pieces of women judged "man" from 3 crops each | split halves inherited the parent's embeddings → re-linked at 1.0 |
| 4 | v1.3 (motion-predicted gap split, box-contained duplicates, vote split needs a box jump, sub-second gender-uncertain not blurred) | 56 | 12 (4 uncertain) | one handoff joined a 100 px far box to a 470 px near one | white-coat woman (0.32 s) skipped because age-only doubt counted as uncertain | gap splits 13 → 2; duplicates found 14 |
| 5 | v1.3 + handoff size guard + age-only doubt still blurs | 58 | 12 (5 uncertain) | none found on the frames/sheets checked | red-coat strip behind reporter (1.4 s); hooded back-view person blurred on a 1:1 vote (may be a man) | current state |

## Measured on this clip (drives the thresholds)

- OSNet osnet_x1_0_msmt17 cosine, different people (confident Woman vs Man pairs, n=406): p50 0.58, p90 0.69, p99 0.76, max 0.83. Same-person pieces across a gap: median 0.61. → body link 0.85 + motion continuity + class agreement.
- ArcFace (buffalo_l) cosine, different-gender people (n=17): max 0.16. → face link 0.45.
- Consecutive crops of one track: p05 0.65 — an appearance-based id-switch detector cannot fire here.
- Qwen2.5-VL-7B on back views / partial crops says "man" often enough that 3 confident votes on a short fragment are not evidence. Face-visible crops are reliable (interviewee 8/8, reporter 8/8).

## Open defects after run 5

1. Partial view behind the foreground person escapes (needs an occlusion re-attachment rule or a second judge).
2. Back-view-only fragments carry weak votes; uncertain ones are blurred by default (escape-safe) and land in review.
3. 4 "None" identities per 12 s (edge slivers, blobs) — harmless, but noise in the review page.

## Speed (RTX PRO 6000)

stage 1 24 fps · stage 2 10 s · stage 3 ~0.4 s per crop, ~600–700 crops → 4–5 min · stage 4 <1 s · stage 5 0.2 min.

# OfflineBlur — team summary (trial of 17 September 2026)

**Goal.** An offline pipeline that pixelates every adult woman in a Dawah-style video (street interviews, Speakers
Corner), keeps audio, and also outputs a white-on-black matte and per-frame masks with identity ids for editors.
Rules in priority order: (1) never blur a non-person or a man, (2) never let a woman escape the blur, (3) masks must be
pixel-accurate, stable and never bleed onto the person next to her. Children ≤ 12 and men are never blurred.

**Hardware.** One RunPod pod, NVIDIA RTX PRO 6000 Blackwell (96 GB). Everything runs on the pod; the Mac only holds code and results.

## Architectures used (v1.4)

| stage | job | model / method | licence |
|---|---|---|---|
| 1 | detect + segment + track every person, every frame | **YOLO11x-seg** (Ultralytics, COCO "person", instance masks at native resolution, inference at 1280–1920 px) + **BoT-SORT** with re-identification (5 s lost-track memory) | AGPL-3.0 (Ultralytics) |
| 2 | cut tracks into single-person segments; describe each | motion-predicted gap split + single-frame hop split; **OSNet x1.0 (MSMT17)** person re-id embeddings via boxmot; **InsightFace SCRFD + ArcFace (buffalo_l)** face embeddings on crops upscaled to 512 px; mask-overlap duplicate flags | OSNet MIT; InsightFace packs non-commercial (replace before commercial release) |
| 3 | judge every crop | **Qwen2.5-VL-7B-Instruct** (open VLM) with a fixed prompt: real person / poster / not a person; man or woman; child or adult + age. Per-crop weights: face visible ×2, sliver 0 (mask fills < 35 % of its box or shorter than max(48 px, 10 % of frame height)) | Apache 2.0 |
| 4 | identities + decisions | classes decided **per segment first**, then union-find linking: mask duplicates → host; face similarity ≥ 0.45 (≥ 2 faces each side; ≥ 0.60 with ≥ 3 faces to cross a gender lean); body similarity ≥ 0.85 + motion continuity + same gender lean; handoff only for lean-less segments on IoU ≥ 0.3. Identity class = pooled vote over all crops (share = mean of weighted and count share). Child only if child answers dominate and median age ≤ 12 | — |
| 5 | render | per-identity mask dilated 2 % of height, unioned over ±1 frame, feathered 6 px, pixelated (block = 10 % of height); gender-uncertain identities blurred only with share ≥ 0.60, weight ≥ 2, ≥ 0.5 s | — |
| 6 | review | self-contained HTML page: one card per identity, uncertain first, crops + votes | — |

Rejected on the way: DINOv2 as a body embedding (v1.0, weak), OSNet links at 0.72 (v1.1, merged strangers: different people
reach 0.83 on night footage), box-containment duplicates (v1.3, attached background people to the near person), face
weight ×3 (v1.3, one answer outvoted three).

Design principle that emerged: **one detector, many frames per person, decide the class before linking**, so a wrong
link can never turn a woman into a man. Every stage writes plain files; stages 4–6 re-run in under a minute.

## Video outputs per clip

Every run produces, under `out/<clip>/render/`: `<clip>_blurred.mp4` (deliverable, audio kept), `<clip>_matte.mp4`
(feathered white-on-black), `<clip>_debug.mp4` (every tracked person outlined with id + class), `masks_rle.jsonl`
(per-frame COCO RLE mask for every human identity, with class and a blurred flag), `render_summary.json`, plus
`review.html` and `crops/` one level up. Compact side-by-side previews (debug | blurred) are in `out/share/`.

| clip | source | resolution / length processed | people | blurred as Woman | false blur | escape | preview file |
|---|---|---|---|---|---|---|---|
| 1. Night street interview | owner-supplied (Ali Dawah) | 1280×720, 12 s | ~25 | 12 identities (interviewee all 300 frames; red-coat, grey-outfit, orange-jacket, pink-bag, stroller women) | none after v1.3 | red-coat woman's strip behind the reporter, 1.4 s | `out/share/clip1_night_street_12s_debug+blurred.mp4` |
| 2. Speakers Corner, Catholic woman | Dailymotion x95zq22 | 640×360, first 240 s | ~40 | 9 identities (interviewee 227.7/240 s as one identity; woman in black; hijab-wearing woman with child) | none after same-day fixes (before: an arm, a blob, a man merged into a woman) | one grey-top figure at 220 s to check; ~1 s of a boy blurred when the mask slid from him to the woman behind | `out/share/clip2_small.mp4` |
| 3. Speakers Corner, Dutchman conversion | Dailymotion x9hqpp8 | 1280×720, first 90 s | ~60 | 45 identities (women crossing behind: prams, veiled woman, blond women, hijab-wearing women) | none after v1.4 (before: a man's head 0.5 s, a backpack person 4.2 s, a handoff onto a man 0.3 s) | blond woman in a blue jacket, 3.8 s (judge tie 4:4, review-flagged) | `out/share/clip3_small.mp4`, `out/share/clip3_dutchman_720p_90s_blurred_only.mp4` (with audio) |

Ali Dawah, the interviewers, the men at the tables and every child checked were never blurred in any clip.

## Timings (RTX PRO 6000)

Stage 1: 24 fps at 720p/1280 px, 67 fps at 360p. Stage 2: 10–30 s per clip. Stage 3 (the judge): ~0.4 s per crop,
~2000 crops per 90–240 s of busy footage → 14–15 min. Render: ~150 fps. Rule of thumb: **~4× real time on busy
Speakers Corner footage, ~1× on a two-person interview.**

## Measured facts behind the thresholds

- OSNet cosine between different people (night crowd): p50 0.58, p99 0.76, max 0.83; same person across a gap: median 0.61.
- ArcFace cosine between different-gender people: max 0.16.
- Qwen2.5-VL-7B on back views and partial crops says "man" often enough that 3 confident votes on a short fragment are not evidence; face-visible crops are reliable (interviewees 22:2, 8:0; interviewers 8:0, 10:0).

## Open items

1. Occlusion slivers (a woman partly hidden behind the foreground person) are judged "man".
2. Judge ties are left unblurred and flagged; a second, non-language judge (MiVOLO V2) is the planned tie-breaker.
3. A mask can slide from a child in front onto the woman behind without a box jump.
4. 360p sources leave ~30 % of background people unjudged; 720p ~7 %. Test HD only.
5. Fragmentation: 2–3 identities per background woman (harmless for the blur, noisy for review).
6. Licences: Ultralytics (AGPL) and InsightFace packs (non-commercial) need replacing or licensing before a commercial release.

## Where everything is

- Code + reports: `OfflineBlur/` (this folder), `offlineblur_v1.4.tar.gz` for redeploying on a new pod (`pod_setup.sh`, then `run.sh <video>`).
- Full pod working folder (all runs, renders, masks, review pages, crops, logs, test clips): `OfflineBlur/pod_archive/`.
- HTML trial report with frames and contact sheets: https://claude.ai/artifact/Q1oaBA5TRCL6vSQyYWALmb
- Per-clip reports and the running log: `OfflineBlur/reports/`.

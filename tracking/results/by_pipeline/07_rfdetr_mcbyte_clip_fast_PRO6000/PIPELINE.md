# 07 · RF-DETR-Seg + McByte + OpenCLIP — fast in-memory version (`tracking/fast_blur.py`), RTX PRO 6000
Same pipeline and decisions as 06; video in → blurred video out in one process, masks kept on the GPU, batch 8.
| | |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell 96 GB, 48 vCPU |
| Speed | **85 Hz** end to end on the 12 s clip (**3.5 s**), **93 Hz** on a 48 s video (12.9 s); 3 repeats, ±1% |
| Per frame | RF-DETR 4.8 ms · CLIP 3.3 ms · decode 3.1 ms (overlapped) · crops 0.9 · blur 0.8 · McByte 0.8 |
| Model load | 6.4–7 s per run (keep warm in a service) |
| Peak GPU | 4.7–5.4 GB |
| Accuracy | 93% of the validated pipeline's women-blur pixels; the 7 label flips were ambiguous back-view/hooded people |
| Faster, accuracy NOT yet checked | 2XL + 5 crops 102 Hz · XLarge 116 Hz · XLarge + 5 crops 130 Hz |

Clip: `ali-dawah-street-interview-source.mp4` — 300 frames, 1280x720, 25 fps, 12 s, ~60 people, night street.
Accuracy uses the 21 hand-labelled people (11 women, 10 men; `../gt_gender_ali_rfdetr_tracks.json`, scorer
`tracking/gender_gt_eval.py`): **women blurred** = average share of each woman's frames where the blur covers at
least half of her; **men false-blur** = the same over the men.

**How to read `labels_masks_ids*.mp4`** (renderer `fast_blur.py --debug-out`): mask FILL colour = track identity
(same colour = same id), box and tag colour = gender decision (magenta woman, cyan man), tag = `#id  gender  P(female)`
(blurred when P(female) >= 0.25). Nothing is blurred in these. Top bar: frame, tracked / women / men counts.
`blur_side_by_side.mp4` = original left, blurred right.
Metrics: `metrics/rep` (speed repeats), `metrics/b8_x264.json`, `metrics/long_*.json`, `metrics/script_baseline`
(the script pipeline on the same pod, ~14 Hz as run).

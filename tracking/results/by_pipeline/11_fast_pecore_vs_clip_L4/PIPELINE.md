# 11 · Fast pipeline, classifier swap — CLIP ViT-L/14-336 → PE-Core-L/14-336, L4

The latest fast structure (`tracking/fast_blur.py`, runner `tracking/run_fast_pecore.sh`) run twice on identical tracks;
only the per-person gender classifier changes.

| | CLIP (run 10 settings) | PE-Core |
|---|---|---|
| Detector + mask | RF-DETR-Seg 2XLarge, fp16, batch 8, person ≥ 0.15 | same |
| Tracker | McByte box, new track ≥ 0.5, buffered IoU 0.5, 2nd round 0.3, lost 4 s | same |
| Classifier | OpenAI CLIP ViT-L/14-336 (`openai`) | Meta PE-Core-L/14-336 (open_clip `PE-Core-L-14-336`, `meta`) |
| Decision | K = 10 masked views ≥ 5 frames apart, P(woman)+P(girl) ≥ 0.25 | same (0.25 not re-tuned for PE-Core) |
| GPU | NVIDIA L4 24 GB, libx264 | same |
| Speed, end to end | **18.8 Hz** (repeat 18.8) | **17.4 Hz** (repeat 17.4) |
| Classifier cost | 16.0 ms/frame | 21.0 ms/frame |
| Peak GPU / load | 4.7 GB / 9.7 s | 6.05 GB / 14.8 s |
| Tracks / labelled women | 48 / 18 | 48 / 17 |

The two agree on 45 of 48 tracks (#7 and #10 woman under CLIP only, #45 woman under PE-Core only).
PE-Core normalises inputs with mean = std = 0.5; `fast_blur.py` now reads each model's own preprocess config.
Not scored against hand labels: the labelled run's track ids could not be rebuilt on this pod (see run 12).

Videos: `blur_original_clip_pecore.mp4` (original | CLIP | PE-Core), `labels_clip_left_pecore_right.mp4`,
and each run full size (`clip_vitl_*`, `pecore_l_*`). Legend as 07.

# 12 · RF-DETR-Seg + EdgeTAM + PE-Core-L, L4

Run 11 (PE-Core) with McByte replaced by **EdgeTAM** (facebookresearch/EdgeTAM, an on-device SAM 2), in
`tracking/fast_blur_edgetam.py` (runner `tracking/run_fast_edgetam.sh`). Judged by eye to be clearly more accurate than
McByte (masks follow people through occlusion and detector misses; fewer identity breaks).

EdgeTAM's video predictor refuses new objects after tracking starts, so the script drives `track_step` per person,
each with its own memory:
1. every live person is propagated from their own memory (mask + object-present score);
2. RF-DETR detections ≥ 0.15 are matched to those masks (mask IoU ≥ 0.3);
3. an unmatched detection ≥ 0.5 less than 50% covered by tracked masks starts a person (box prompt);
4. of two tracks overlapping ≥ 0.7 the younger is dropped; a track unconfirmed by any detection for 4 s ends.
The blur uses EdgeTAM's mask on every frame the person is present; classifier views come only from detection-confirmed frames.

| | |
|---|---|
| GPU | NVIDIA L4 24 GB, bf16 EdgeTAM, libx264 |
| Speed | **3.6 Hz** end to end (12 s clip), **3.2 Hz** on the 48 s version |
| Per frame | EdgeTAM 218 ms (image encoder 16 · per-person steps 200 · new tracks 1.6 · matching 0.6) · RF-DETR 26 · PE-Core 21 · blur + encode 8 |
| Per person | 9.3 ms per EdgeTAM step; batching people gives nothing on the L4 (GPU compute bound) |
| Peak GPU / load | 6.6 GB (7.1 GB on 48 s) / 15.5 s |
| Tracks / labelled women | 54 / 22, ~16 people tracked per frame |
| Memory | EdgeTAM memory flat (~480 entries); the pipeline keeps frames, masks and views until the end: +0.5 MB/frame GPU, +2.7 MB/frame RAM |

EdgeTAM upstream crashes with more than one person per batch (`.view` on an expanded tensor in `sam2/modeling/perceiver.py`);
the pod copy uses `.reshape` (same result). Not scored against hand labels: rebuilding the labelled run's track ids on a new pod
gave 59 ids instead of 60 (7138 vs 7171 detections), so the old labels no longer point at the same people;
`tracking/track_strips.py` makes blind contact sheets for relabelling.

Videos: `blur_original_mcbyte_edgetam.mp4` (original | McByte + PE-Core | EdgeTAM + PE-Core),
`labels_mcbyte_left_edgetam_right.mp4`, `edgetam_pecore_blur.mp4`, `edgetam_pecore_labels_masks_ids.mp4`. Legend as 07.
Latency of every L4 pipeline: `tracking/reports/l4_pipeline_latency.html`.

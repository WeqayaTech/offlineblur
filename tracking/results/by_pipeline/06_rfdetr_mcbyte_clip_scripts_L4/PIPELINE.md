# 06 · RF-DETR-Seg + McByte + OpenCLIP — script version, L4
| Role | Component |
|---|---|
| Detector + mask | RF-DETR-Seg **2XLarge** (rfdetr 1.10.1, 768 px, FP16, COCO `person`, threshold 0.15) |
| Tracker | McByte box overlap, high/activation 0.5, lost buffer 30 |
| Classifier | OpenCLIP **ViT-L/14-336 (openai)** on the 10 best masked crops per track; woman if P(woman)+P(girl) ≥ 0.25 |
| GPU | NVIDIA L4 22 GB |
| Speed | detector 44 ms + McByte 3 ms + CLIP 41 ms (27 ms per crop) = 88 ms/frame · **11.4 Hz** compute (8.3 Hz as run) |
| Peak GPU | 3.1 GB |
| Accuracy | women blurred **0.696** (= oracle labels), men false-blur **0.029**, 0 women never blurred; 60 tracks |

Clip: `ali-dawah-street-interview-source.mp4` — 300 frames, 1280x720, 25 fps, 12 s, ~60 people, night street.
Accuracy uses the 21 hand-labelled people (11 women, 10 men; `../gt_gender_ali_rfdetr_tracks.json`, scorer
`tracking/gender_gt_eval.py`): **women blurred** = average share of each woman's frames where the blur covers at
least half of her; **men false-blur** = the same over the men.

**How to read `labels_masks_ids.mp4`** (renderer `render_gender.py`): masks are FILLED by gender concept —
magenta = woman, blue = man — with a `prompt:id` tag. Thin grey outlines are the reference `person` layer (SAM 3.1
tracked); a grey outline with no fill is a person the pipeline did not gender in that frame (a potential miss).
`blur_side_by_side.mp4` = original left, blurred right.
Metrics: `metrics/rfdetr_2XLarge`, `metrics/rfdetr_mcbyte`, `metrics/rfdetr_mcbyte_clip`, `metrics/rfdetr_mcb_mask`
(McByte masks: 274 ms/frame, +2 points), `metrics/rfclip_vitl336_masked_b0.25`, `metrics/rfclip_siglip_masked`
(SigLIP comparison, worse: 0.482).

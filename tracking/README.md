# tracking/ — detection + tracking bake-off (the new core)

Per [../ROADMAP.md](../ROADMAP.md), detection + tracking is the foundation and "what to blur" is a
layer on top. This module compares tracker candidates on our footage and outputs `tracks.jsonl`
(one box per person per frame with a stable id) — the interface everything downstream consumes.

## The bake-off

MOTRv3 has **no public code or weights** (unanswered release request on the MOTRv2 repo, no
repository anywhere), so it cannot be deployed. Its runnable modern equivalent is **MOTIP** (CVPR
2025, MCG-NJU), which we test against the objective-analysis favourite, **BoT-SORT + strong ReID**,
and a third, architecturally very different candidate, **SAMURAI** (SAM 2.1 + motion-aware memory):

| candidate | what | adapter |
|---|---|---|
| **MOTIP** | end-to-end transformer (ID-prediction), the MOTRv3 stand-in. Ships DanceTrack/SportsMOT weights only — no street/MOT17 checkpoint, so we run the DanceTrack weights (dancer domain gap). | `adapters/motip_track.py` |
| **BoT-SORT + ReID** | YOLO11x detector + boxmot BoT-SORT with camera-motion compensation and an appearance model, run fresh every frame (tracking-by-detection). | `adapters/botsort_track.py` |
| **SAMURAI** | [yangchris11/samurai](https://github.com/yangchris11/samurai) — SAM 2.1's video predictor with a Kalman-filter motion model. Zero-shot, no training, but architecturally a **single-object VOT tracker**, not a multi-object one — see below for what that costs. | `adapters/samurai_track.py` |

All three write the same `tracks.jsonl`, so `compare.py` scores any two of them head to head.

### SAMURAI's architecture, and what it took to get a fair multi-person run

SAMURAI has no detector of its own and no native multi-object mode — using it for crowd tracking is
an adaptation, not its designed use case. Three real constraints surfaced building the adapter, in the
order we hit them:

1. **The motion-aware Kalman filter is singleton state on the model**, not batched:
   `self.kf_mean` / `self.kf_covariance` / `self.stable_frames` / `self.frame_cnt` are set once in
   `SAM2Base.__init__` and never reset. Seeding >1 object on one `inference_state` crashes inside
   `_forward_sam_heads` (`ious[0][best_iou_inds]` — a per-batch-index tensor used where a scalar is
   expected). **Fix:** one full single-object `init_state` → `propagate_in_video` pass per person,
   resetting those four attributes on the model between people.
2. **A frame-0-only seed misses everyone who enters later.** The adapter originally seeded once from
   YOLO on frame 0 and propagated the whole clip. Measured on a 12 s/300-frame street clip: the gap
   between raw YOLO detections and SAMURAI's boxes grew from 2/frame at the start to 8–12/frame by
   mid-clip and stayed there — **43 % of all detections across the clip belonged to people SAMURAI
   never tracked.** **Fix:** re-run YOLO every `--rescan-every` frames (default 25), IoU-match against
   every already-tracked object's most recent box (with a short lookback for momentary occlusion), and
   give any unmatched detection its own fresh single-object pass.
3. **A late-seeded object still can't start mid-video on the full frame sequence.** SAMURAI's
   motion-aware memory scoring (`_prepare_memory_conditioned_features` in `sam2_base.py`) selects
   recent frames with a hardcoded `range(frame_idx - 1, 1, -1)` over `non_cond_frame_outputs` — it
   assumes the tracked object's history starts at absolute frame 0. Seeding a new object at frame 25 on
   the full 300-frame sequence makes it look up frame 25's entry there, but frame 25 is the
   *conditioning* frame (stored in `cond_frame_outputs`), not a non-cond one → `KeyError: 25`. **Fix:**
   give each new entrant its own frame sequence starting at local index 0 — a directory of symlinks to
   the frames from its entry point onward (`make_frame_slice`) — then remap the returned local frame
   indices back to absolute ones when writing `tracks.jsonl`.

With all three fixes, on that same clip SAMURAI went from 17 ids / 3854 boxes (frame-0-seed only) to
**42 ids / 7201 boxes**. It still can't match BoT-SORT's ids (109, vs SAMURAI's 42) because SAMURAI
never subdivides an identity — no re-detection-driven fragmentation — while BoT-SORT's per-frame
detection is directly gated by its own track-birth threshold (next section). Cost: SAMURAI is far more
expensive than BoT-SORT — one SAM 2 forward+memory pass per person per frame it's alive, vs one YOLO
pass per frame total for BoT-SORT.

### BoT-SORT's hidden track-birth threshold

boxmot's `BotSort` gates track **birth** separately from the detector conf you pass it:
`new_track_thresh` defaults to **0.6** and `track_high_thresh` to **0.5** — both well above the
`conf=0.25` used to query YOLO. A detection at, say, 0.35 confidence (real, visible in the raw-detector
video) would pass the detector but never be allowed to start a new track id. Measured: of 685
detections on the same clip that had no matching BoT-SORT box that frame, **91.5 % scored below
0.6**. Fix: `botsort_track.py` now exposes `--new-track-thresh` / `--track-high-thresh`, defaulted to
0.25 to match the detector.

### Detector recall check (`detect_bakeoff.py`)

Both trackers depend on the same YOLO11x person detector — if it misses someone, no tracker downstream
can ever find them. `detect_bakeoff.py` compares raw detections only (no tracking/ids) between the
`baseline` config both adapters use (single pass, imgsz 1280, conf 0.25) and a `tiled` config
(full-frame pass + overlapping 2×2 tile passes at native tile resolution, conf 0.10, merged with NMS).
On the same clip, tiled roughly doubled recall (18.2 → 38.0 boxes/frame average) but was not adopted —
eyeballed side by side, baseline was judged solid enough and tiled's extra boxes weren't worth its
~5x compute cost and higher false-positive risk for this footage.

    python3 detect_bakeoff.py --frames-meta <seq>/frames_meta.json --out <out_dir> [--yolo yolo11x.pt]

## Pod

Offline batch, inference only — both fit well under 24 GB. **A40 48 GB or L40S 48 GB** (Ampere/Ada,
cheaper than A100, room for SAM 2 masks later). A100 80 GB also fine. **Not Blackwell** (RTX PRO 6000
/ 5090 / B200): the deformable-attention CUDA op MOTIP needs fails on the stock torch there. 50 GB
container disk. Keep everything on the container disk (`/root/tracking`, `/root/hf`).

## Run

```bash
bash tracking/pod_setup_tracking.sh                 # clone MOTIP, build its op, install boxmot, fetch weights
bash tracking/get_clips.sh                          # re-download the 6-clip corpus (set IA_1/IA_2 for archive.org)
bash tracking/run_bakeoff.sh /root/tracking/videos/<clip>.mp4          # MOTIP vs BoT-SORT + side-by-side video
bash tracking/run_samurai_bakeoff.sh /root/tracking/videos/<clip>.mp4  # SAMURAI vs BoT-SORT + side-by-side video
```

`run_samurai_bakeoff.sh` needs the SAMURAI repo and a SAM 2.1 checkpoint set up separately (clone
[yangchris11/samurai](https://github.com/yangchris11/samurai) `--recursive`, `pip install -e sam2/`,
download a `sam2.1_hiera_*.pt` checkpoint) — see `SAMURAI_REPO` / `SAMURAI_CKPT` / `SAMURAI_MODEL` env
vars at the top of the script. Both bake-off scripts also accept `IMGSZ` / `CONF` / `REID` /
`SAMURAI_CONF` / `GPU` env overrides.

Per clip you get `out/<clip>/<tracker>/tracks.jsonl` for each tracker run, a
`compare/<clip>_compare.mp4` (first tracker left, second right) and `compare/<clip>_metrics.json`.

## Metrics

These clips have **no ground truth**, so `compare.py` reports GT-free proxies: track-span median
(continuity), short-track fraction and late births (fragmentation), boxes/frame and simultaneous
count (recall), ids-with-gaps (occlusion handling). True ID switches / IDF1 / HOTA need labels — for
real numbers, run both on a labeled set (MOT17-val or MOT20) and score with TrackEval; that leg is
not wired here yet. The decisive check for our footage is the side-by-side video.

## Notes / things to adjust on first run

- MOTIP's `submit` mode reads `datasets/DanceTrack/test/<seq>/img1`; if it expects a `seqmap` file for
  the split, drop one listing the single seq (the adapter otherwise mirrors the MeMOTR/MOTRv2 layout).
- The deformable-op path in `pod_setup_tracking.sh` assumes `MOTIP/models/ops`; adjust if the repo
  moved it.
- `get_clips.sh` needs the archive.org item URLs (`IA_1`, `IA_2`) for the two daytime clips; the four
  Dailymotion ids are hard-coded.

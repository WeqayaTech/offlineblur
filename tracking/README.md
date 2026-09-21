# tracking/ — detection + tracking bake-off (the new core)

Per [../ROADMAP.md](../ROADMAP.md), detection + tracking is the foundation and "what to blur" is a
layer on top. This module compares tracker candidates on our footage and outputs `tracks.jsonl`
(one box per person per frame with a stable id) — the interface everything downstream consumes.

## The bake-off

MOTRv3 has **no public code or weights** (unanswered release request on the MOTRv2 repo, no
repository anywhere), so it cannot be deployed. Its runnable modern equivalent is **MOTIP** (CVPR
2025, MCG-NJU), which we test against the objective-analysis favourite, **BoT-SORT + strong ReID**:

| candidate | what | adapter |
|---|---|---|
| **MOTIP** | end-to-end transformer (ID-prediction), the MOTRv3 stand-in. Ships DanceTrack/SportsMOT weights only — no street/MOT17 checkpoint, so we run the DanceTrack weights (dancer domain gap). | `adapters/motip_track.py` |
| **BoT-SORT + CLIP-ReID** | YOLO11x detector + boxmot BoT-SORT with camera-motion compensation and a CLIP appearance model (far above OSNet on night crowds). | `adapters/botsort_track.py` |

Both write the same `tracks.jsonl`, so `compare.py` scores them head to head.

## Pod

Offline batch, inference only — both fit well under 24 GB. **A40 48 GB or L40S 48 GB** (Ampere/Ada,
cheaper than A100, room for SAM 2 masks later). A100 80 GB also fine. **Not Blackwell** (RTX PRO 6000
/ 5090 / B200): the deformable-attention CUDA op MOTIP needs fails on the stock torch there. 50 GB
container disk. Keep everything on the container disk (`/root/tracking`, `/root/hf`).

## Run

```bash
bash tracking/pod_setup_tracking.sh                 # clone MOTIP, build its op, install boxmot, fetch weights
bash tracking/get_clips.sh                          # re-download the 6-clip corpus (set IA_1/IA_2 for archive.org)
bash tracking/run_bakeoff.sh /root/tracking/videos/<clip>.mp4     # MOTIP vs BoT-SORT + side-by-side video
```

Per clip you get `out/<clip>/motip/tracks.jsonl`, `out/<clip>/botsort/tracks.jsonl`, a
`compare/<clip>_compare.mp4` (MOTIP left, BoT-SORT right) and `compare/<clip>_metrics.json`.

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

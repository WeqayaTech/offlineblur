# OfflineBlur — direction

Updated 2026-09-18.

## The shift

Priority is now **person detection + tracking as the standalone core**. Everything that decides
*what* to blur (gender, age, or a user's manual pick) becomes a thin, swappable layer on top of a
solid track. Rationale: a wrong blur or an escape is almost always a tracking failure (a missed
person, an id switch, a lost occlusion) rather than a classifier failure. Get one clean, continuous
identity per person and the rest is a selection problem.

## Two layers

1. **Core — detection + tracking (the focus).** Input: a video. Output: `tracks.jsonl`, one box per
   person per frame carrying a stable identity id that survives occlusion. This is the piece to make
   reliable and measurable first. Nothing above it can be better than this is.

2. **Selector — what to blur (later, pluggable).** Input: `tracks.jsonl` (+ optional per-crop
   attributes). Output: the set of identity ids to blur. Interchangeable implementations:
   - **gender** — the v2 trained head or the v3 zero-shot VLM judge (already built).
   - **manual** — a person reviews the identity contact sheet and picks ids (or picks "blur everyone
     except these"). No model needed.
   - **future** — age, specific-person match, or a mix.

   The renderer (`blur.py`) already takes a set of ids, so any selector that writes that set works
   unchanged.

## What "tracking is solved" means (acceptance)

Measured on the six-clip corpus in `reports/`:
- Every person visible for ≥ N frames carries exactly one id from entry to exit.
- No id switch across a short occlusion (person walks behind the reporter and comes back).
- No id shared by two different people.
- Detection recall on small/background people (< ~60 px) high enough that no visible person is missed.
Report per clip: id switches, fragmentations (one person split into several ids), merges (two people
sharing one id), and misses. These are standard MOT metrics (IDF1, MOTA, ID switches) plus a manual
eyeball of the debug video.

## Status of the pieces (already in the repo)

- Tracking adapters: MOTRv2 (default) and MeMOTR, driven unmodified — `v2/stp/tracker.py`.
- Detection for proposals / masks: YOLO11x and YOLO11x-seg.
- Debug video + per-identity contact sheet for eyeballing tracks — `v2/stp/render.py`, `v2/stp/sheet.py`.
- Selectors that exist: v2 trained gender head, v3 zero-shot VLM gender judge.
- Blur renderer that consumes a chosen id set — `v2/stp/blur.py`.

## Next steps (tracking-first)

1. Decide the primary tracker to invest in (transformer line vs a lean detector+tracker) — see the
   open question below.
2. Build a tracking evaluation harness: run the chosen tracker on the six-clip corpus, dump the MOT
   metrics above, and render debug videos for the eyeball pass.
3. Fix the top tracking failures that harness surfaces (id switches through occlusion, small-person
   recall) before touching classification again.
4. Only then: firm up the selector interface and add the manual "user picks ids" selector.

## Service (2026-09-23)

The deployment shape is fixed in [service/PLAN.md](service/PLAN.md) and uses the terms defined
there. In short: the GPU side runs SAM 3 once per video over a fixed label set (`woman`, `man`,
`child`, with `person` as the control prompt) and returns a metadata bundle of canonical
identities, masks and per-label coverage. The device does selection, live blur and export. The
server never renders video.

The "selector" layer above is what the bundle's coverages feed: the manual selector is the device's
identity gallery, the gender selector is a label toggle plus a coverage threshold. The current
limitation stands as measured there: tracking is strong, label-as-selector has a 25 % escape rate on
the trial clip, so the device always shows a review group and never auto-exports from a label
toggle alone. The research harness (`tracking/run_sam3_gender.sh` and its report and render
scripts) remains the way we measure that number; it is not part of the service.

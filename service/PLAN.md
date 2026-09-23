# SAM 3 short-video service — deployment plan

Branch `sam3/deploy-plan`, revised 2026-09-23. Plan only; no service code exists yet.

## 1. Approach in one paragraph

The GPU side runs SAM 3 **once** per video over a **fixed label set** and returns a **metadata
bundle**: one canonical identity per person, per-frame masks, and per-identity label coverage.
The **device** does everything after that: choosing labels, blurring live, changing the choice
with no round trip, and exporting the final video. The server never renders video and keeps the
upload only while the job runs.

```
device                                          GPU side (RunPod)
------                                          -----------------
upload video  ---------------------------->     ffmpeg -> frames (720p cap, stride)
                                                SAM 3, all labels in one pass, chunked
                                                stitch + relink -> canonical identities
              <----------------------------     bundle (manifest + masks + thumbnails)
                                                delete upload and frames
keep the original locally
decode bundle in a Web Worker
play video, blur selected identities in a shader, live
toggle labels / identities -> instant
export blurred mp4 locally
```

Why this shape: one GPU pass ever per video (the label set is fixed, so no user choice can
trigger a re-run, and the deterministic result is cached by content hash); no server rendering;
selection is a lookup-table change in a shader; and the only copy of the blurred output is the one
the device makes.

## 2. Terminology

These words are used the same way in every section, and in `ROADMAP.md`.

| term | meaning |
|---|---|
| **label** | a noun-phrase prompt in the server's fixed set (`woman`, `man`, `child`). A label is what the user can choose to blur. |
| **control prompt** | `person`. Run alongside the labels as the recall floor and as the base of the identity space. Not user-selectable. |
| **canonical identity** | one id per physical person in the video, built from the control prompt and any label objects overlapping it. The unit the device blurs. |
| **coverage** | for one identity and one label, the fraction of that identity's frames the label's mask covered. Carried in the bundle; thresholded on the device. |
| **verdict** | the device-side classification of an identity from its coverages: `woman`, `man`, `child`, `conflict` (two labels both above threshold), `flicker` (a label present but below threshold), `ungendered` (no label). |
| **escape** | an identity whose verdict is `flicker` or `ungendered`. If that person should have been blurred, they ship unblurred. The only error that cannot be undone after export. |
| **bundle** | the zip the GPU side returns: `manifest.json`, `masks.bin`, `thumbs/`. |
| **job** | one uploaded video processed once. Keyed by content hash. |
| **device** | the customer's browser or app. Where selection, blur and export happen. |
| **GPU side** | the worker that runs SAM 3. Stateless per job. |
| **research harness** | `tracking/run_sam3_gender.sh`, `sam3_gender_report.py`, `render_gender.py`. Evaluation tools that render videos on the pod. They are how we measure; they are not part of the service. |

## 3. Current limitation, stated once

**Tracking is the strong part; label-as-selector is not yet reliable enough to blur
automatically.** Measured on the trial clip (`ali-dawah-street-interview-source.mp4`, 300 frames,
1280x720, labels `woman, man, child`, control `person`):

| measure | value |
|---|---|
| `person` identities | 69 (vs 118 McByte, 109 BoT-SORT on the same clip) |
| identities with no label at all | 12 % |
| identities with a label below threshold (flicker) | 13 % |
| **escape rate** (the two rows above) | **25 %** |
| identities claimed by both `woman` and `man` | 10 % |
| `woman` observations also claimed by `man` | 28 % |

Consequences that every later section respects:

- The bundle carries **raw coverages**, not a verdict, so the device can move the threshold and
  the user can see the effect. Nothing about the label decision is baked in server-side.
- The device **never auto-exports** from a label toggle alone. The default view pre-selects the
  chosen label and puts `conflict`, `flicker` and `ungendered` identities in a "review these"
  group with their own toggles. The user confirms before export.
- The knobs that move the escape rate (`new_det_thresh`, `score_threshold_detection`, richer
  phrases than a bare `woman`) are **server config**, tuned with the research harness on the
  six-clip corpus, and recorded in the bundle's manifest so a bundle is always attributable to the
  config that made it.

Everything else measured about SAM 3 that the design relies on:

| run (trial clip, 720p) | ids | s/frame | peak VRAM | GPU |
|---|---|---|---|---|
| 4 prompts, chunk 100 | 69 | ~1.9 | 11.1 GB flat | RTX PRO 4500 32 GB |
| person only, chunk 100 | 61 | 0.86 | 5.6 GB flat | RTX PRO 4500 |
| person only, one session, 900 frames, pruned | 88 | 1.55 | 17.4 GB | 24 GB card |
| first frame (warm-up) | | ~18 s | | |

SAM 3 is deterministic (golden tests, content-hash cache are safe). Chunks are independent sessions
stitched afterwards (chunks can fan out across workers, section 8).

## 4. The fixed label set

Server config, not a per-job option. Initial set: labels `woman`, `man`, `child`; control
`person`. All run in one propagation pass sharing vision features, so an extra label costs little
GPU time but adds mask output. Adding a label is a config change plus a golden-test re-run. The
device receives whatever set the server was configured with and shows those as toggles.

**Canonical identities.** Prompts produce separate object ids (a `woman` object and a `person`
object on the same pixels). The bundle writer collapses them using the per-pixel mask-IoU rollup
already in `sam3_gender_report.py`: `person` ids are the base; each label object is assigned to
the `person` id it overlaps; a label object overlapping no `person` id becomes its own canonical
identity, so a `woman` detection is never dropped because the control missed her. Per identity,
per label, the bundle records coverage and mean score.

## 5. The bundle

One `bundle.zip` per job, served by a signed URL.

**`manifest.json`**

```
{
  "version": 1,
  "video": {"fps": 29.97, "width": 1280, "height": 720, "n_frames": 900, "duration_s": 30.03,
            "content_sha256": "..."},
  "processing": {"stride": 2, "mask_width": 640, "mask_height": 360, "chunk_frames": 100,
                 "model_snapshot": "3c879f39...", "image": "sam3-worker:2026-09-23-5f0b3c3",
                 "labels": ["woman","man","child"], "control": "person",
                 "new_det_thresh": 0.7, "score_threshold_detection": 0.5,
                 "seconds": 142.3, "gpu": "H100"},
  "identities": [
    {"id": 7, "first_frame": 0, "last_frame": 611, "n_obs": 298,
     "labels": {"woman": {"coverage": 0.91, "score": 0.83},
                "man":   {"coverage": 0.04, "score": 0.52},
                "child": {"coverage": 0.00, "score": 0.0}},
     "thumb": "thumbs/7.jpg", "box_first": [412, 88, 610, 690]}
  ],
  "frames": [{"f": 0, "ids": [7, 12], "boxes": [[412,88,610,690],[...]]}, ...]
}
```

`frames` carries boxes only, one entry per processed frame, tens of KB for 900 frames. It lets the
device draw a timeline of who is on screen before the masks are decoded.

**`masks.bin`** the per-pixel data, the only large part, laid out for decode speed on a device:

- Stored at `mask_width x mask_height` (default 640x360, half the processed frame). A blur mask
  does not need full resolution once it is dilated a few pixels on the device.
- Per processed frame: `uint16 n_ids`, then per identity: `uint16 id`, `uint16 n_runs`,
  `uint16[] runs` (alternating skip/fill run lengths over the row-major grid, COCO RLE order).
  Little-endian.
- Whole file gzip-compressed. Browsers decompress it natively with `DecompressionStream`; no codec
  dependency on the device.
- Frames skipped by `stride` are not stored; the device reuses the nearest processed frame's masks.

Size estimate from the trial clip (about 22 canonical masks per frame, ~600 runs each at 640x360,
2 bytes per run, gzip ~2.5x): roughly 0.3 MB per processed second, so a 30 s clip at stride 2 is
about 5 MB and at stride 1 about 10 MB. These are estimates; M0 measures them on the corpus. If
they are off by more than 2x, the fallback is a lossless label-map video (one 8-bit frame per
processed frame, pixel value = identity id), which compresses better across frames but needs a
codec the device can decode.

**`thumbs/<id>.jpg`** one crop per identity, from the frame with its highest score.

`tracks.jsonl` and `masks.jsonl` are not in the bundle. The worker still writes them to scratch so
the research harness and the golden test can read them; an internal debug flag can include them.

## 6. API

Small on purpose. All under `/v1`, API key in the `Authorization` header.

**`POST /jobs`** multipart `video`, or JSON `{"url": ...}`. No prompt options. Knobs:
`max_seconds` (default 30, hard cap 60) and `stride` (default 2, allowed 1 to 3), clamped
server-side. Returns `{"job_id", "status", "cached": bool, "eta_seconds_est"}`. If the content
hash already has a bundle, `status` is `done` immediately and no GPU work happens.

**`GET /jobs/{id}`** `{"status": queued|running|done|failed, "progress": {"frame", "n_frames",
"sec_per_frame", "eta_seconds"}, "bundle_url", "error"}`. The device polls every 2 s. There is no
per-frame event stream: the device cannot use partial masks before stitching, and dropping it
removes a moving part.

**`DELETE /jobs/{id}`** removes the bundle and the cache entry for that hash.

The upload is deleted by the worker as soon as the bundle is written. Bundles expire after 24 h.

## 7. Packaging and serving: containers, and which kind

**Yes, containers, and exactly two images.** The alternative, a hand-set-up pod with a Python
environment, is what the research harness uses and it is the wrong unit for a service: it is not
reproducible (`pod_setup_sam3.sh` already has to guard against three silent install traps), it
cannot be autoscaled, and there is no way to say which code and weights produced a given bundle.

**`sam3-worker` image (GPU).**

- Base: a PyTorch 2.7+ / CUDA 12.8 image so Blackwell and Hopper cards both work.
- Pinned Python deps from a lock file: `transformers` with `Sam3VideoModel`,
  `kernels>=0.16,<0.17`, `pycocotools`, `opencv-python-headless`, `ffmpeg` from apt.
- **Weights baked in** from the transformers snapshot already on the RunPod volume (3.4 GB). The
  image is about 10 GB. Baking beats mounting a network volume for two reasons: the image is a
  complete, versioned artefact (tag = git sha + model snapshot hash, recorded in every manifest),
  and RunPod caches images on its hosts so only the first cold start on a host pays the pull.
  A network volume keeps the image small but adds a mount dependency and a network read of 3.4 GB
  on every cold start.
- Container start runs the arch check and bf16 matmul from `pod_setup_sam3.sh`, loads the model
  once, then enters the job loop.
- One entrypoint, two modes selected by env: `runpod` (Serverless handler: receives a job payload,
  writes the bundle to object storage, returns its key) and `queue` (pulls jobs from Redis, for a
  persistent pod). Same code path underneath.
- No inference server framework (Triton, TorchServe, vLLM-style) is needed. Those serve many small
  requests against one loaded model. This workload is one long batch job per video with a model
  that is already resident, so a plain handler loop is the right shape and one fewer thing to run.

**`sam3-api` image (CPU).** FastAPI, ffprobe validation, upload to object storage, job enqueue,
poll endpoint, signed URLs, TTL sweeper, and the static device app. Small, stateless, runs anywhere
(a RunPod CPU pod, Fly, a VPS). Topology A runs both containers on one GPU pod with a compose file;
Topology B runs `sam3-api` on a CPU host and `sam3-worker` on RunPod Serverless.

**Build and release.** GitHub Actions builds both images on every merge to `main`, runs the golden
test inside `sam3-worker` on a RunPod GPU runner, and pushes to a registry only if the golden
bundle is byte-identical. The manifest's `image` field is how any bundle is traced back to the
exact build.

## 8. Scaling to many videos

Two independent axes, and they are solved differently.

**Axis 1: many videos at once (throughput).** Horizontal. Each job is one video and one worker
processes one job at a time; the queue holds the rest.

- **Topology B autoscaling.** RunPod Serverless scales workers on queue depth: min 0 (or 1 warm
  during working hours), max N set by budget. A worker that finishes a job pulls the next; idle
  workers scale down after a timeout. Cost is per second of GPU actually used.
- **Packing per GPU.** Peak VRAM per job at 4 prompts is 11 GB (chunked) with state on the host,
  so an 80 GB card can run 3 to 4 worker processes at once, each with its own model copy. Whether
  concurrent jobs slow each other (host RAM and PCIe for the CPU-resident state) is an M0
  measurement; if the slowdown is under 30 % this is the cheapest throughput lever.
- **Fairness and backpressure.** Per-API-key concurrent-job limit (default 2) so one client
  cannot fill the queue; `POST /jobs` returns 429 with a `Retry-After` when queue depth exceeds a
  threshold; jobs are FIFO within a key.
- **Idempotency.** The content-hash cache means N uploads of the same video are one job. Salted
  per API key so clients cannot probe each other's uploads.

**Axis 2: one video fast (latency).** Chunk map-reduce. SAM 3 chunks are independent sessions
stitched afterwards, so a 900-frame job becomes 9 chunk tasks (100 frames plus 10 overlap) that
run on 9 workers at once, followed by a CPU reduce that runs the existing `stitch` and `relink` in
order and then the bundle writer. Exact with respect to sequential chunked output. Wall time drops
from ~13 min to roughly one chunk's time plus reduce, about 2 to 3 min at the measured speed.
Chunk tasks are just jobs on the same queue, so axis 1's autoscaling serves axis 2 too.

**Capacity arithmetic to size N.** At the measured ~1 s/frame and stride 2, a 30 s clip is ~450
processed frames, ~7.5 min of GPU per job sequentially. One worker does about 8 clips per hour;
with chunk fan-out the same GPU-minutes are spent but each clip returns in ~3 min. Ten workers give
~80 clips per hour. These numbers move with the GPU tier chosen in M0 (an H100 is expected 2 to 3x
faster than the measured cards) and should be re-derived from `service/BENCH.md`.

**What does not scale and is out of scope.** Real-time or live input. The plan is for uploaded
short clips; a frame-in streaming mode would use SAM 3's streaming path, which disables its
hotstart de-duplication and raises false positives, and is not offered.

## 9. GPU worker internals

Package the existing adapter as `service/worker/` without forking it:

- **Model singleton.** `build_model` loads once per process; `run` takes the loaded model.
- **Fixed prompt set from config.** Labels plus control, every job.
- **Frame extraction.** ffmpeg to JPEG on container disk with the 720p cap and stride in the
  filter graph. Frames and the upload are deleted at job end, success or failure.
- **Bundle writer.** New module: canonical-identity rollup (lifted from `sam3_gender_report.py`),
  per-identity coverages, thumbnails, `masks.bin` at half resolution, zip. Input is the adapter's
  `masks.jsonl`, so the adapter itself is untouched.
- **Limits.** `max_num_objects` 200, job timeout `n_frames * 3 s`, then fail with the scratch
  output kept for debugging.

## 10. Device app

A web app first (desktop and phone browsers). The bundle is client-agnostic, so a native app can
consume it later without server changes.

1. **Upload and wait.** Drag/drop or URL, progress bar from the poll. The original stays in memory
   or an Origin Private File System handle.
2. **Bundle decode.** A Web Worker streams `masks.bin` through `DecompressionStream`, parses runs
   into typed arrays, and expands frames to an 8-bit id map (pixel = identity id, 0 = none) lazily
   in a ring buffer of ~90 frames around the playhead (~20 MB). Ids above 255 are remapped per
   frame.
3. **Live blur.** `<video>` drawn into a WebGL canvas. Per frame: upload the id map as a texture
   plus a 256-entry lookup texture of "selected or not". The fragment shader samples the video and
   the id map at the same UV and, where selected, samples a pixelated version instead, dilating
   the mask a few texels. Toggling a label rewrites the lookup: no decode, no re-render.
4. **Selection.** Label toggles that select every identity whose coverage exceeds a threshold
   slider; an identity gallery (thumbnail, coverages, frame span) with per-identity overrides; a
   timeline of who is on screen; the "review these" group from section 3.
5. **Export.** WebCodecs: `VideoDecoder` on the original, the same shader offscreen,
   `VideoEncoder` (H.264, hardware where available), mux with the original audio into mp4 with a
   small muxer library. Fallback without WebCodecs: `MediaRecorder` from the canvas (WebM).
   Nothing is uploaded during export.

## 11. Privacy and security

- TLS; API key per client; per-key rate and concurrency limits (section 8).
- ffprobe validation first: size cap 200 MB, duration cap, container allowlist, fixed temp paths,
  ffmpeg with a timeout. Filenames never touch a shell.
- The video exists on the GPU side only for the duration of the job. Bundles carry masks and
  thumbnails, no full frames, and expire after 24 h. `DELETE` is immediate.
- No frame content in logs; job id, counters, timings and errors only.
- Content-hash cache salted per API key.

## 12. Observability

Per job in `manifest.processing` and as metrics: queue wait, cold or warm load, sec/frame, peak
VRAM, ids per prompt, bundle size, cache hit, chunk count. `/healthz` on the API, worker heartbeat.
Alerts on cold start over 120 s, sec/frame over 3, bundle size over 30 MB, any failure, queue depth
over the backpressure threshold for more than 10 min.

## 13. Testing and acceptance

- **Golden regression.** The trial clip yields byte-identical `tracks.jsonl` and `masks.jsonl` to
  the committed golden run (62 ids / 7131 observations for `person`) on every image build, and the
  bundle writer yields a byte-identical `bundle.zip` from that. The build does not publish
  otherwise.
- **Chunk equivalence.** Map-reduce output equals sequential chunked output exactly.
- **Stride A/B.** Ids and escape rate at stride 1, 2, 3 on the trial clip; confirm stride 2 and
  that half-resolution masks dilated by 3 px cover the full-resolution masks.
- **Bundle size.** `masks.bin` on the six-clip corpus; decide RLE binary vs label-map video.
- **Concurrency.** s/frame with 1, 2, 4 worker processes on one 80 GB card.
- **Device.** Decode a 30 s bundle on a mid-range Android phone in under 5 s; play at native fps
  with blur on; toggle a label with no visible hitch; export a 30 s clip in under 2x real time on a
  laptop; the mp4 opens in QuickTime, VLC and a phone gallery with audio intact.
- **API.** Bad container, over-length, over-size, unknown job, delete, cache hit, salted hash, 429
  under backpressure.
- **Acceptance loop for the demo URL.** Upload a 20 s phone clip, get the bundle, see the gallery,
  toggle `woman`, scrub with blur live, review the flagged identities, export, play the export.
  Under 5 min end to end.

## 14. Milestones

| # | deliverable | done when |
|---|---|---|
| M0 | Benchmark GPU tiers at 720p, 4 prompts, stride 1 and 2; concurrency per card; `masks.bin` size on the corpus | numbers in `service/BENCH.md`; GPU tier, stride default, mask format, workers-per-card chosen |
| M1 | `sam3-worker` image: model singleton, ffmpeg ingest, fixed prompts, bundle writer, baked weights, golden test through to `bundle.zip`, CI build | CI produces the golden bundle from the image on a GPU runner |
| M2 | Device app: bundle decoder, WebGL live blur, label and identity selection, timeline, review group | plays the golden bundle over the trial clip with live toggles, desktop and phone |
| M3 | Topology A: `sam3-api` image, in-process queue, poll endpoint, compose file, device app served from the pod | acceptance loop passes on a single-pod URL |
| M4 | Device export via WebCodecs with audio, MediaRecorder fallback | export checks in section 13 pass |
| M5 | Topology B: Serverless handler mode, Redis, object storage, content-hash cache, TTL sweeper, autoscaling, backpressure | acceptance loop against the split deployment; scales to zero; 429 under load |
| M6 | Chunk map-reduce | 30 s clip bundle under 3 min; output identical to sequential |

M0 and M2 run in parallel (M2 needs only a bundle from an existing adapter run). M1 before M3, M3
before M5, M5 before M6. M4 is independent once M2 exists.

## 15. Decisions needed from the owner

1. Initial label set: `woman, man, child` with `person` as control, or a different list.
2. Clip limits for v1: 30 s default, 60 s hard cap, 720p, stride 2.
3. Web app only, or a native app consuming the same bundle from day one.
4. Bundle retention: 24 h server-side, or delete on first download.
5. Object storage provider for Topology B (Cloudflare R2 vs RunPod S3-compatible).
6. Budget ceiling for max workers, which sets N in section 8.

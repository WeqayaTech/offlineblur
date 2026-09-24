# SAM 3 short-video service — deployment plan

Branch `sam3/deploy-plan`, revised 2026-09-23. Plan only; no service code exists yet.

## 1. Approach in one paragraph

The GPU side runs SAM 3 **once** per video over a **fixed label set** and returns a **metadata
bundle**: one canonical identity per person, per-frame masks, and per-identity label coverage.
The **device** does everything after that: choosing labels, blurring live, changing the choice
with no round trip, and exporting the final video. The server never renders video and keeps the
upload only while the job runs.

The device is the **app team's existing app**. They integrate against the API contract in
[`API.md`](API.md), which is the document to hand them. This plan builds the service behind that
contract plus a small **test console** for the owner to exercise the service end to end before
any API key is issued to the team.

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
| **device** | the app team's app on the customer's phone or desktop. Where selection, blur and export happen. |
| **test console** | a single page served by the API, owner-only, that uploads, polls, and previews a bundle with live blur. Exists to test the service, not for customers. |
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
  "job_id": "job_01J8...",
  "video": {"source": {"width": 1920, "height": 1080},
            "width": 1280, "height": 720, "fps": 29.97, "n_frames": 900, "duration_s": 30.03,
            "content_sha256": "..."},
  "processing": {"stride": 2, "mask_width": 640, "mask_height": 360, "chunk_frames": 100,
                 "model_snapshot": "3c879f39...", "config_version": "2026-09-23-5f0b3c3",
                 "labels": ["woman","man","child"], "control": "person",
                 "new_det_thresh": 0.7, "score_threshold_detection": 0.5,
                 "seconds": 142.3, "gpu": "H100"},
  "masks": [{"from_f": 0, "to_f": 898, "path": "masks.bin"}],
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
- Per processed frame: `uint32 f`, `uint16 n_ids`, then per identity: `uint16 id`,
  `uint32 n_runs`, `uint16[] runs` (alternating skip/fill run lengths over the row-major grid,
  COCO RLE order; a 65535 run continues into the next run of the same kind). Little-endian. The
  exact layout and a reference decoder are in `API.md`, which is the source of truth for the
  format.
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

The full contract, with request and response shapes, error codes, the bundle format and a
reference decoder, is [`API.md`](API.md). That file is what the app team receives. Summary:

| endpoint | purpose |
|---|---|
| `GET /v1/labels` | the configured label set, control prompt, config version and limits, so the app builds its toggles from the server |
| `POST /v1/jobs` | upload (multipart) or submit a URL; optional `max_seconds`, `stride`, `webhook_url`; returns `job_id`, or `done` at once on a cache hit |
| `GET /v1/jobs/{id}` | status, progress, signed `bundle_url` when done, structured error when failed |
| `DELETE /v1/jobs/{id}` | delete the bundle and forget the cache entry |
| webhook | one signed POST on done or failed, for apps that prefer not to poll |
| `GET /v1/healthz` | liveness |

Design choices: no prompt options (fixed label set); no per-frame event stream, since the device
cannot use masks before stitching; polling at 2 s or a webhook, the app's choice; API keys per app
environment with a concurrent-job limit. The upload is deleted by the worker as soon as the bundle
is written. Bundles expire after 24 h.

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

## 10. Device side: the team's app, and the owner's test console

**The team's app** is the customer-facing device. What it must do with a bundle (decode, choose,
blur live, export on the device) is specified for them in `API.md` under "The bundle", including
the recommended threshold logic, the review group, the dilation, and the ring-buffer decode. The
service makes no assumption about their stack beyond "can gunzip and draw pixels".

**The test console** is for the owner only, to exercise the service before the team gets a key.
It is one static HTML page served by `sam3-api` at `/console`, gated by the same API key typed
into the page (kept in `localStorage`, never in the page source). It is deliberately minimal and
uses no framework:

1. Upload a file or paste a URL, pick `stride`, submit. Shows the job id, status and progress
   from the poll, and the raw JSON of every response so the owner sees exactly what the app team
   will see.
2. On `done`, downloads the bundle, decodes it in a Web Worker with the same reference decoder
   that `API.md` publishes (so the console is also the decoder's test).
3. Identity gallery from the manifest: thumbnail, coverages per label, frame span, verdict at the
   current threshold, and the review group. Label toggles and a threshold slider.
4. Live preview: the uploaded video in a `<video>` element drawn to a 2D canvas with the selected
   identities pixelated from the id map. 2D canvas, not WebGL, because this is a test surface and
   correctness matters more than frame rate.
5. Buttons for `DELETE` and for downloading the bundle and the manifest.

No export. The console proves the service and the bundle format; exporting video is the app's job
and is specified in `API.md`.

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
- **Bundle decoder.** The reference decoder in `API.md` reproduces the worker's `masks.jsonl`
  masks pixel for pixel from `masks.bin` (round-trip test, run in CI in Node).
- **Test console.** Decodes the golden bundle, shows 62 identities, blur follows the toggles and
  the threshold slider, `DELETE` returns 404 afterwards.
- **API.** Bad container, over-length, over-size, unknown job, delete, cache hit, salted hash, 429
  under backpressure, webhook delivered with a valid signature, `GET /labels` matches the config.
- **Acceptance loop before the team gets a key.** In the console: upload a 20 s phone clip, watch
  progress, get the bundle, see the gallery, toggle `woman`, scrub with blur live, review the
  flagged identities, delete. Under 5 min end to end. Then hand the team a dev key, the sample
  bundle, the clip it came from, and `API.md`.

## 14. Milestones

| # | deliverable | done when |
|---|---|---|
| M0 | Benchmark GPU tiers at 720p, 4 prompts, stride 1 and 2; concurrency per card; `masks.bin` size on the corpus | numbers in `service/BENCH.md`; GPU tier, stride default, mask format, workers-per-card chosen |
| M1 | `sam3-worker` image: model singleton, ffmpeg ingest, fixed prompts, bundle writer, baked weights, golden test through to `bundle.zip`, CI build. Produces the **sample bundle** for the team. | CI produces the golden bundle from the image on a GPU runner |
| M2 | Test console: reference decoder (shared with `API.md`), gallery, toggles, threshold, 2D-canvas live blur | plays the golden bundle over the trial clip with live toggles |
| M3 | Topology A: `sam3-api` image, `/labels`, jobs endpoints, in-process queue, API keys, compose file, console served at `/console` | acceptance loop passes on a single-pod URL |
| M4 | **Team hand-off**: dev key with limits, webhook with signature, `API.md` checked line by line against the live service, sample bundle and clip delivered | the team's app decodes the sample bundle and completes one live job against the dev key |
| M5 | Topology B: Serverless handler mode, Redis, object storage, content-hash cache, TTL sweeper, autoscaling, backpressure | acceptance loop against the split deployment; scales to zero; 429 under load |
| M6 | Chunk map-reduce | 30 s clip bundle under 3 min; output identical to sequential |

M0 and M2 run in parallel (M2 needs only a bundle from an existing adapter run). M1 before M3, M3
before M4, M4 before M5, M5 before M6. The team can start integrating from the sample bundle as
soon as M1 produces it, before the API is live.

## 15. Growth path: longer videos, bigger uploads, streamed input

v1 is scoped to short clips because of GPU time, not because of the model or the contract. This
section records what changes when the scope grows, so nothing in v1 is built in a way that has to
be torn out. Every step below is **additive** to the v1 contract.

**The model is not the limit.** A single SAM 3 session already completes 900 frames with memory
pruning on, and chunked sessions have flat memory at any length. Wall time is the limit: at the
measured ~1 s/frame and stride 2, a 20-minute 1080p recording (a typical 2 GB phone file) is
~18,000 processed frames, about 5 hours on one worker, or about 15 minutes fanned out over 20
workers with chunk map-reduce (section 8). So "large video" is a cost and parallelism decision per
job, and the chunk fan-out built in M6 is the mechanism. The `max_seconds` cap becomes a per-key
setting rather than a global one.

**Upload transport.** A 2 GB multipart POST through the API is the wrong path: slow, not
resumable, and it ties up an API worker. The change is a presigned direct-to-storage upload:
`POST /v1/uploads` returns an upload id and presigned part URLs, the app PUTs parts straight to
object storage (resumable, parallel), then `POST /v1/jobs {"upload_id": ...}`. The v1 JSON form of
`POST /jobs` already separates "where the video is" from "run the job", so this is a new source
type, not a new endpoint shape. Topology B (object storage in front of the worker) is the
prerequisite, which is why v1 is designed around it even though the demo runs on one pod.

**Bundle size.** Masks grow linearly: ~0.3 MB per processed second, so a 20-minute recording is
~180 MB of masks. The v1 manifest therefore lists masks as **segments** from day one
(`manifest.masks: [{from_f, to_f, path}]`), with exactly one entry for short clips. Long jobs
write one segment per minute or so, and the device fetches only the segments around the playhead.
The `masks.bin` layout is per-frame self-delimiting, so splitting it into files changes nothing in
the decoder.

**Progressive results.** With chunk map-reduce, the reducer finalises identities in frame order,
so segments 0 to k can be published while later chunks are still running. `GET /jobs/{id}` gains
`segments_ready` and the device can start reviewing the first minute of a long recording before
the last minute is processed. Identities in a published segment never change; a person who
reappears later gets linked to their earlier id by the stitch, which only ever assigns ids forward.

**Append mode (record now, process as you go).** The middle ground between "upload the whole
file" and "stream frames": the app uploads a recording in pieces (for example 10 s mp4 segments)
while it is still recording, and processing starts on piece 1 while piece 2 uploads. Server side
each piece is a chunk and the boundary is stitched exactly like a chunk boundary today, using the
tail of the previous piece as the overlap. Contract: `POST /jobs {"mode": "append"}`,
`PUT /jobs/{id}/pieces/{n}`, `POST /jobs/{id}/finish`. This is the realistic form of "frame by
frame upload" and it needs no model change.

**True frame-by-frame or live input.** Two problems, one on each side. Transport: sending frames
instead of a video is 50 to 100x more bytes (a JPEG per frame versus H.264) and gains nothing,
because the worker decodes video to frames on the GPU host anyway. Model: SAM 3's streaming path
disables its hotstart de-duplication (more false positives) and, more decisively, the measured
~1 s/frame is 30x too slow for real time at 30 fps. So live input waits on a faster model or GPU
generation. When it arrives, the contract is a separate `WebSocket /v1/streams` session that emits
the same per-frame mask records the bundle uses, with provisional identities (no post-hoc stitch).
Nothing in v1 has to change for that to be added.

**What this means for v1 decisions.** Build Topology B with object storage and a real queue even
for a demo-scale service; put `masks` as a segment list in the manifest now; keep `POST /jobs`
source-agnostic; treat `max_seconds` as a per-key limit. None of these cost anything at short-clip
scale and each removes a future breaking change.

## 16. Decisions needed from the owner

1. Initial label set: `woman, man, child` with `person` as control, or a different list.
2. Clip limits for v1: 30 s default, 60 s hard cap, 720p, stride 2.
3. Whether the team's app polls or wants the webhook; both are in the contract, but knowing which
   one they will use first decides what M3 must have working.
4. Bundle retention: 24 h server-side, or delete on first download.
5. Object storage provider for Topology B (Cloudflare R2 vs RunPod S3-compatible).
6. Budget ceiling for max workers, which sets N in section 8.
7. Console access: API key typed into the page (proposed), or put it behind basic auth as well.

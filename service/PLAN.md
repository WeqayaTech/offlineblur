# SAM 3 short-video service — deployment plan

Branch `sam3/deploy-plan`, written 2026-09-22. This is the plan only; no service code exists yet.

## 1. What we are shipping

A URL where a user uploads a short video and gets back, for every person in it, a stable identity,
a per-frame box and per-pixel mask, and a label (`woman` / `man` / `person`, plus a per-identity
verdict). Optionally a rendered labels video and a blurred video. The GPU side is the SAM 3
multi-prompt adapter that already exists in `tracking/adapters/sam3_track.py`; this plan wraps it
in a service, it does not change how it tracks.

What the model does today, measured on the trial clip (`ali-dawah-street-interview-source.mp4`,
300 frames, 1280x720), so the service is planned around real numbers:

| run | ids | s/frame | peak VRAM | GPU |
|---|---|---|---|---|
| 4 prompts (woman,man,person,child), chunk 100 | 69 person ids | ~1.9 | 11.1 GB flat | RTX PRO 4500 32 GB |
| person only, chunk 100 | 61 | 0.86 (4.3 min) | 5.6 GB flat | RTX PRO 4500 |
| person only, one session, 900 frames, pruned | 88 | 1.55 (23.2 min) | 17.4 GB | 24 GB card |
| first frame (warm-up) | | ~18 s | | |

Two facts from those runs shape the design:

- **SAM 3 is deterministic.** Same input, same output byte for byte. That gives us a golden-file
  regression test and makes any run-to-run diff a real bug.
- **Chunks are independent sessions.** Identity is stitched afterwards from overlap masks. That
  means chunks can run on different GPUs in parallel and be reduced on CPU, which is the only
  lever that makes a 30-second clip come back in minutes rather than tens of minutes.

Known limit that the service must not hide: gender-as-selector is **not shippable as an automatic
blur decision yet**. On the trial clip 25 % of `person` identities got no gender concept at all
(escapes) and 28 % of `woman` observations were also claimed by `man`. So v1 returns tracking plus
labels with confidence and lets the client review and pick the blur set. Auto-blur is a flag the
client can turn on, off by default.

## 2. Frame-by-frame or whole video? Whole video, with results streamed back per frame

The question in the brief was whether the client sends frames one at a time or the whole file.
Recommendation: **the client uploads the whole file; the server streams results back frame by
frame as they are produced.**

Why not frame-by-frame input:

- SAM 3's streaming mode disables the hotstart de-duplication heuristics, and the model card says
  that raises false positives. Every measurement we have is on pre-loaded video.
- The adapter already loads a chunk of frames into memory before propagating. Feeding one frame at
  a time buys nothing on the GPU and adds a round trip per frame over the network.
- Upload is one request, resumable, easy to size-limit and validate.

Why streaming results out still matters: the propagation loop yields one frame at a time, so the
client can draw boxes on a canvas as the job runs and the user sees progress within ~20 s of
submitting instead of staring at a spinner for minutes. A frame-in websocket mode is kept as a
phase-5 option for live/webcam input, with the false-positive cost measured before it is offered.

## 3. Architecture

```
browser (static page)
   | 1. POST /v1/jobs  (multipart video, or {"url": ...})
   v
API (FastAPI, CPU, always on)
   | validates with ffprobe, caps size/length, writes video to object storage
   | enqueues job
   v
queue (Redis)  ---->  GPU worker(s) (RunPod)
                        | ffmpeg -> frames on container disk (720p cap, optional fps stride)
                        | sam3 chunk sessions  (model loaded once per worker process)
                        | per-frame progress + observations -> Redis pub/sub
                        | stitch + relink -> tracks.jsonl / masks.jsonl / identities.json
                        | optional: labels.mp4, blur.mp4
                        v
                      object storage (results, signed URLs, TTL)
   ^
   | 2. GET /v1/jobs/{id}          status + progress
   | 3. GET /v1/jobs/{id}/events   SSE, one event per frame
   | 4. GET /v1/jobs/{id}/result   URLs to artefacts
browser draws overlays live, then shows the identity gallery for review
```

Two topologies, in the order we build them:

**Topology A, single pod (demo URL, week 1).** One RunPod GPU pod runs FastAPI, an in-process
worker queue, and serves the static page. Storage is the container disk. No Redis, no object
storage. This is the fastest path to a working URL and is enough for internal review.

**Topology B, split (production).** API on a cheap always-on CPU host, GPU on RunPod Serverless
with a Docker image that has the SAM 3 weights baked in, Redis for queue and progress, S3-compatible
object storage (Cloudflare R2 or RunPod's) for uploads and results. Scales to zero, pays per second
of GPU. Cold start is the cost: image pull plus model load plus ~18 s first frame. Keep one worker
warm during working hours, or accept ~60 to 90 s on the first job.

The worker code is identical in both. Topology is a config choice, not a rewrite.

## 4. API contract

All endpoints under `/v1`, API key in the `Authorization` header, JSON unless stated.

**`POST /jobs`** multipart `video` file, or JSON `{"url": "..."}` for a public link. Options:

| field | default | meaning |
|---|---|---|
| `prompts` | `["woman","man","person"]` | concept prompts, all in one pass |
| `blur_prompt` | `"woman"` | which prompt is the candidate blur set |
| `max_seconds` | 30 | hard cap on processed duration |
| `max_long_side` | 1280 | frames downscaled to this before SAM 3 |
| `fps_stride` | 1 | process every N-th frame; skipped frames inherit the nearest mask |
| `chunk_frames` | 100 | SAM 3 session length |
| `new_det_thresh` | model default 0.7 | the escape/precision knob, exposed on purpose |
| `outputs` | `["tracks","masks","identities"]` | add `"labels_video"`, `"blur_video"` |
| `auto_blur` | false | render blur video from `blur_prompt` without review |

Returns `{"job_id", "status": "queued", "n_frames_est", "eta_seconds_est"}`. Rejects with 4xx on
bad container, over-length, over-size, or non-video content.

**`GET /jobs/{id}`** `{"status": queued|running|done|failed, "progress": {"frame", "n_frames",
"sec_per_frame", "eta_seconds", "ids_per_prompt"}, "error"}`.

**`GET /jobs/{id}/events`** Server-Sent Events. One `frame` event per processed frame carrying
`{f, obs: [{tid, prompt, score, box}]}` (boxes only; masks are too large to stream). A final `done`
event with the result URLs. Reconnect with `Last-Event-ID` resumes from that frame.

**`GET /jobs/{id}/result`** signed URLs plus inline summary:

- `tracks.jsonl` `{f, tid, box, score, prompt}` (existing schema, unchanged)
- `masks.jsonl` `{f, tid, prompt, score, rle}` COCO RLE at processed resolution (existing schema)
- `identities.json` per identity: `tid, prompt, verdict (woman|man|conflict|flicker|ungendered),
  first_frame, last_frame, n_obs, mean_score, thumbnail_url` — this is the per-identity **label**
  the brief asks for, produced by `sam3_gender_report.py` logic
- `metrics.json` the existing GT-free report (escape rate, conflict rate, ids per prompt)
- `labels.mp4`, `blur.mp4`, `sbs.mp4` when requested
- `meta.json` fps, size, stride, chunking, worker GPU, seconds, model snapshot hash

**`POST /jobs/{id}/render`** `{"blur_tids": [..]}` renders a blur video for a chosen id set. This
is the manual selector from the roadmap. CPU only, reads `masks.jsonl`, does not touch the GPU.

**`DELETE /jobs/{id}`** removes upload and all artefacts immediately.

Keeping the on-disk schema identical to what the adapter writes today means `render_gender.py`,
`sam3_gender_report.py` and `sam3_chunk_eval.py` keep working on service output without changes.

## 5. GPU worker

Package the existing code as `service/worker/` without forking it:

- **Model singleton.** `build_model` currently runs per call. Load once per process at worker
  start, keep it on the GPU, pass it into `run`. Saves the load time on every job.
- **Progress callback.** `run()` prints per 25 frames. Add an `on_frame(af, entries)` callback and
  have the CLI keep its prints, so the service publishes to Redis and the CLI behaves as before.
- **Frame extraction.** ffmpeg to JPEG on container disk (never the network volume), with the
  720p cap and stride applied in the ffmpeg filter, not in Python. Frames are deleted with the job.
- **Chunk map-reduce (Topology B only).** Emit one serverless task per chunk (100 frames plus 10
  overlap), each returning its observations with masks. A CPU reducer runs the existing `stitch`
  and `relink` over chunk outputs in order and assigns global ids. Wall time for a 900-frame clip
  drops from ~13 min sequential to roughly one chunk's time plus reduce, about 2 to 3 min. This is
  exact with respect to the current chunked algorithm because sessions are already independent.
- **Weights.** Bake the transformers snapshot (`model.safetensors`, 3.4 GB) into the image. No
  gated download at runtime, no HF token in the worker. The snapshot already on the RunPod volume
  is the source for the image build.
- **Environment.** Same rules as `pod_setup_sam3.sh`: PyTorch 2.7+ / CUDA 12.8 image so Blackwell
  cards work, `transformers` new enough for `Sam3VideoModel`, `kernels>=0.16,<0.17`,
  `HF_HUB_DISABLE_XET=1`, state device `cpu`, `expandable_segments:True`. The arch check and bf16
  matmul from the setup script run at container start and fail fast.
- **Limits.** `max_num_objects` capped (200) so a crowd cannot grow memory without bound. Job
  timeout of `n_frames * 3 s` after which the job fails with a partial result kept.

## 6. Client

A single static page, no framework needed for v1:

1. Upload (drag and drop or URL), options panel with the defaults above, submit.
2. Live view: the video element plus a canvas. Boxes arrive over SSE and are drawn at the matching
   frame time, colour by prompt (woman magenta, man blue, person outline only, matching
   `render_gender.py` so the two views agree).
3. Identity gallery: one card per identity with thumbnail, prompt, verdict, frame span, and a
   checkbox pre-ticked for `woman` verdicts and unticked for everything else. `conflict` and
   `flicker` cards are flagged so the reviewer looks at them.
4. Render button posts the ticked ids to `/render` and shows the blurred video with a download link.
5. Delete button.

Served by the API process in Topology A. In Topology B it is a static bundle on any host with the
API base URL as config.

## 7. Throughput budget and the levers

At the measured ~0.9 to 1.9 s/frame, a 30 s clip at 30 fps (900 frames) takes 13 to 28 min on one
worker. That is not an interactive service. The levers, in the order we pull them:

1. **Downscale to 720p** on ingest. Already the measured resolution; 1080p input would be slower.
2. **fps stride 2 or 3.** Halves or thirds the frame count. Masks on skipped frames are copied from
   the nearest processed frame; for blur that is fine once the mask is dilated a few pixels. Needs
   one A/B on the trial clip to confirm ids do not fragment at 10 to 15 fps.
3. **Fewer prompts.** `woman,person` is enough for the blur use case; `man` is the diagnostic
   contrast. Two prompts instead of four should be near the person-only 0.86 s/frame.
4. **Chunk map-reduce** across serverless workers (section 5). This is the big one.
5. **Faster GPU.** All numbers are from a RTX PRO 4500 or a 24 GB card. An H100 is expected to be
   2 to 3x faster; measure in milestone 0 before choosing the serverless GPU tier.

Target after levers 1 to 4: a 30 s clip returns in under 3 minutes wall time with first overlays on
screen within 30 s.

## 8. Privacy, security, retention

The input is video of real people, so:

- TLS everywhere; API key per client; per-key rate limit and concurrent-job limit.
- Validate the upload with ffprobe before anything else. Size cap (200 MB), duration cap
  (`max_seconds`), container allowlist. Filenames are never interpolated into shell commands;
  ffmpeg gets a fixed temp path and a timeout.
- Uploads and artefacts have a TTL (default 24 h) and are deleted by a sweeper. `DELETE` is
  immediate. Frames on the worker are removed at job end, success or failure.
- No frame content in logs. Logs carry job id, frame counters, timings, and errors only.
- Signed, expiring URLs for every artefact. Nothing is world-readable.

## 9. Observability

Per job, recorded in `meta.json` and exported as metrics: queue wait, model load (cold or warm),
sec/frame, peak VRAM, ids per prompt, escape rate, conflict rate, chunk count, stitched and
relinked counts. `/healthz` on the API and a worker heartbeat in Redis. Alert on cold start over
120 s, sec/frame over 3, and any job failure.

## 10. Testing and acceptance

- **Golden regression.** The trial clip at 300 frames must produce byte-identical `tracks.jsonl`
  and `masks.jsonl` to the committed golden run (62 ids / 7131 observations for `person`). Because
  SAM 3 is deterministic, any diff is a real change. Runs on every image build.
- **Chunk equivalence.** Map-reduce output must equal sequential chunked output exactly.
- **Stride A/B.** Ids and escape rate at stride 1 vs 2 vs 3 on the trial clip; pick the default.
- **API tests.** Bad container, over-length, over-size, unknown job, delete, SSE resume.
- **Load test.** 5 concurrent 30 s jobs on Topology B; measure queue wait and per-job wall time.
- **Acceptance for the demo URL.** Upload a 20 s phone clip, see first boxes within 30 s, get the
  identity gallery, tick ids, download a blurred video. Whole loop under 5 minutes.

## 11. Milestones

| # | deliverable | done when |
|---|---|---|
| M0 | Benchmark on the candidate serverless GPU (H100 and one cheaper tier): sec/frame, VRAM, load time, at 720p with 2 and 4 prompts, stride 1 and 2 | numbers table committed to `service/BENCH.md`; GPU tier and stride default chosen |
| M1 | `service/worker/`: model singleton, `on_frame` callback, ffmpeg ingest, job runner CLI, Dockerfile with baked weights, golden test | `docker run ... job.json` reproduces the golden output on a pod |
| M2 | Topology A: FastAPI, in-process queue, SSE, static client, identity gallery, render endpoint | a URL on a single pod passes the acceptance loop in section 10 |
| M3 | Topology B: RunPod Serverless handler, Redis, object storage, signed URLs, TTL sweeper, API on a CPU host | same acceptance loop against the split deployment; scales to zero |
| M4 | Chunk map-reduce across serverless workers | 900-frame clip under 3 min wall, output identical to sequential |
| M5 (optional) | Frame-in websocket mode for live input | false-positive delta vs batch measured and documented before exposure |

M0 to M2 are sequential. M3 and M4 can proceed in parallel once M2 is up.

## 12. Decisions needed from the owner

1. Clip limits for v1: 30 s and 720p as proposed, or longer.
2. Audience: internal review tool (API key, no accounts) or public (needs accounts, quotas,
   abuse handling). The plan assumes internal.
3. GPU tier for serverless after M0 numbers: fastest, or cheapest that meets the 3-minute target.
4. Whether `auto_blur` should exist at all in v1 given the 25 % escape rate, or whether every job
   goes through the review gallery.
5. Object storage provider for Topology B (Cloudflare R2 vs RunPod S3-compatible).

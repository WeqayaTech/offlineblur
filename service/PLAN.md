# SAM 3 short-video service — deployment plan

Branch `sam3/deploy-plan`, revised 2026-09-23. Plan only; no service code exists yet.

## 1. The split

The GPU does one thing: run SAM 3 over the uploaded video with a **fixed, preconfigured set of
labels**, once, and return **metadata only**. Everything after that happens on the customer's
device: choosing which labels to blur, previewing the blur in real time, changing the choice
without any round trip, and exporting the final video.

```
customer device                                 GPU side (RunPod)
--------------                                  -----------------
upload video  ---------------------------->     ffmpeg -> frames (720p cap, stride)
                                                SAM 3, all labels in one pass, chunked
                                                stitch + relink -> canonical identities
              <----------------------------     metadata bundle (identities + masks)
                                                delete the video and frames
keep the original locally
decode bundle in a Web Worker
play video, blur selected ids in a WebGL shader, live
toggle labels / ids -> instant
export blurred mp4 locally (WebCodecs)
```

Why this is the efficient shape:

- **One GPU pass per video, ever.** The label set is fixed, so nothing about the user's choice
  can trigger a re-run. The same upload twice returns the cached bundle (SAM 3 is deterministic,
  so the bundle is keyed by content hash and reused).
- **No server rendering.** No labels video, no blur video, no side-by-side, no render endpoint.
  The GPU worker stops the moment the metadata is written. The measured pipeline spends real time
  in `render_gender.py`; that time moves to the device, where it is spread across playback.
- **Real-time selection.** Blur = "pixels whose identity id is in the selected set." Changing a
  label toggles a set of ids. That is a lookup-table change in a shader, not a re-render.
- **Privacy.** The server holds the video only while the job runs, then deletes it. It never
  produces or stores a blurred output. The original never leaves the device except for the one
  upload, and the result that gets shared is made on the device.

What the model does today, measured on the trial clip (300 frames, 1280x720), so the sizes and
times below are grounded:

| run | ids | s/frame | peak VRAM | GPU |
|---|---|---|---|---|
| 4 prompts (woman,man,person,child), chunk 100 | 69 person ids | ~1.9 | 11.1 GB flat | RTX PRO 4500 32 GB |
| person only, chunk 100 | 61 | 0.86 (4.3 min) | 5.6 GB flat | RTX PRO 4500 |
| person only, one session, 900 frames, pruned | 88 | 1.55 (23.2 min) | 17.4 GB | 24 GB card |
| first frame (warm-up) | | ~18 s | | |

Two facts from those runs carry into the design: SAM 3 is deterministic (golden tests, content-hash
cache), and chunks are independent sessions stitched afterwards (chunks can fan out across GPUs).

## 2. The preconfigured label set

Server config, not a per-job option. Initial set, all run in one propagation pass sharing vision
features, so an extra prompt costs little GPU time but does add mask output:

| prompt | role in the bundle |
|---|---|
| `person` | recall control and the canonical identity space: every physical person gets one id from this prompt |
| `woman` | label |
| `man` | label |
| `child` | label |

Adding a label later (for example `face`, or a richer phrase like `adult woman`) is a config change
plus a re-run of the golden test. The client never asks for prompts; it receives whatever the
server was configured with and shows those as toggles.

**Canonical identities.** Prompts produce separate object ids (a `woman` object and a `person`
object on the same pixels). The bundle collapses them to one id per physical person using the
per-pixel mask-IoU rollup that `sam3_gender_report.py` already does: `person` ids are the base,
each label prompt's objects are assigned to the `person` id they overlap, and any label object that
overlaps no `person` id becomes its own canonical id (so a `woman` detection is never dropped
because the recall prompt missed her). Per canonical id the bundle carries, for every label, the
fraction of that id's frames the label covered and the mean score. The verdict
(`woman | man | child | conflict | flicker | ungendered`) is computed on the device from those
fractions with a threshold the user can move, not baked in server-side.

Known limit the client must surface, not hide: on the trial clip 25 % of `person` identities got
no gender label at all and 28 % of `woman` observations were also claimed by `man`. So the default
client view pre-selects the chosen label but shows `conflict`, `flicker` and `ungendered`
identities as "review these" with their own toggles.

## 3. The metadata bundle

One file per job, `bundle.zip`, produced by the worker and served by a signed URL. Contents:

**`manifest.json`**

```
{
  "version": 1,
  "video": {"fps": 29.97, "width": 1280, "height": 720, "n_frames": 900, "duration_s": 30.03,
            "content_sha256": "..."},
  "processing": {"stride": 2, "mask_width": 640, "mask_height": 360, "chunk_frames": 100,
                 "model_snapshot": "3c879f39...", "labels": ["woman","man","child"],
                 "control": "person", "seconds": 142.3, "gpu": "H100"},
  "identities": [
    {"id": 7, "first_frame": 0, "last_frame": 611, "n_obs": 298,
     "labels": {"woman": {"frac": 0.91, "score": 0.83},
                "man":   {"frac": 0.04, "score": 0.52},
                "child": {"frac": 0.00, "score": 0.0}},
     "thumb": "thumbs/7.jpg", "box_first": [412, 88, 610, 690]}
  ],
  "frames": [{"f": 0, "ids": [7, 12], "boxes": [[412,88,610,690],[...]]}, ...]
}
```

`frames` carries boxes only, one entry per processed frame. It is small (tens of KB for 900
frames) and lets the client draw a timeline of who is on screen before the masks are decoded.

**`masks.bin`** the per-pixel data, the only large part. Format chosen for decode speed on a
device, not for readability:

- Masks are stored at `mask_width x mask_height` (default 640x360, half the processed frame). A
  blur mask does not need full resolution once it is dilated a few pixels on the device.
- Per processed frame: `uint16 n_ids`, then per id: `uint16 id`, `uint16 n_runs`, `uint16[]
  runs` (alternating skip/fill run lengths over the row-major grid, COCO RLE order). All
  little-endian.
- Whole file gzip-compressed. Browsers decompress it natively with `DecompressionStream`, so the
  client has no codec dependency.
- Frames skipped by `stride` are not stored; the client reuses the nearest processed frame's
  masks, which is why stride exists.

Size estimate from the trial clip (about 22 canonical masks per frame, ~600 runs each at 640x360,
2 bytes per run, gzip ~2.5x): roughly 0.3 MB per processed second, so a 30 s clip at stride 2 is
about 5 MB and at stride 1 about 10 MB. These are estimates to be measured in M0; if they are off
by more than 2x, the alternative is a lossless label-map video (one 8-bit frame per processed frame
where pixel value = identity id) which compresses better across frames but needs a codec the
device can decode, so it is second choice.

**`thumbs/<id>.jpg`** one crop per identity for the review gallery, from the frame with its
highest score.

**`tracks.jsonl` and `masks.jsonl`** are not in the bundle. The worker still writes them to its
scratch dir so `sam3_chunk_eval.py` and the golden test can read them, and a debug flag can add
them to the bundle for internal runs.

## 4. API

Deliberately small. All under `/v1`, API key in the `Authorization` header.

**`POST /jobs`** multipart `video`, or JSON `{"url": ...}`. No prompt options. The only knobs
are `max_seconds` (default 30, hard cap 60) and `stride` (default 2, allowed 1 to 3), both
clamped server-side. Returns `{"job_id", "status", "cached": bool, "eta_seconds_est"}`. If the
content hash already has a bundle, `status` is `done` immediately and no GPU work happens.

**`GET /jobs/{id}`** `{"status": queued|running|done|failed, "progress": {"frame", "n_frames",
"sec_per_frame", "eta_seconds"}, "bundle_url", "error"}`. The client polls this every 2 s; there
is no per-frame event stream, because the client cannot do anything useful with partial masks
before the whole bundle is stitched, and dropping SSE removes a moving part.

**`DELETE /jobs/{id}`** removes the bundle and the cache entry for that hash.

The video itself is deleted by the worker as soon as the bundle is written, whether or not the
client ever downloads it. Bundles expire after 24 h.

## 5. GPU worker

Package the existing adapter as `service/worker/` without forking it:

- **Model singleton.** `build_model` loads once per worker process; `run` takes the loaded model.
- **Fixed prompt set from config.** `run` is called with the server's label list plus the control
  prompt, every job, no exceptions.
- **Frame extraction.** ffmpeg to JPEG on container disk with the 720p cap and stride in the
  ffmpeg filter graph. Frames and the upload are deleted at job end, success or failure.
- **Bundle writer.** New module: rollup to canonical ids (lifting the matching from
  `sam3_gender_report.py`), per-id label fractions, thumbnails, `masks.bin` at half resolution,
  zip. The adapter's `masks.jsonl` is the input, so the adapter itself is untouched.
- **Chunk map-reduce (Topology B).** One serverless task per 100-frame chunk plus 10 overlap; a
  CPU reducer runs the existing `stitch` and `relink` over chunk outputs in order, then the bundle
  writer. Exact with respect to the current chunked algorithm because sessions are already
  independent. This is what turns a 900-frame clip from ~13 min sequential into roughly one
  chunk's time plus reduce, about 2 to 3 min.
- **Weights baked into the image** from the transformers snapshot already on the RunPod volume
  (3.4 GB). No gated download, no token at runtime.
- **Environment** as in `pod_setup_sam3.sh`: PyTorch 2.7+ / CUDA 12.8 image, `transformers` with
  `Sam3VideoModel`, `kernels>=0.16,<0.17`, `HF_HUB_DISABLE_XET=1`, state device `cpu`,
  `expandable_segments:True`, arch check and bf16 matmul at container start.
- **Limits.** `max_num_objects` 200, job timeout `n_frames * 3 s`.

Two topologies, built in order. **A:** one GPU pod runs the API, an in-process queue, the worker,
and serves the client page; container disk only. Fastest path to a URL. **B:** API on a cheap CPU
host, Redis queue, S3-compatible object storage for uploads and bundles, RunPod Serverless workers,
content-hash cache in Redis. Scales to zero; cold start (image pull, model load, ~18 s first frame)
is the cost, mitigated by one warm worker during working hours. The worker code is the same.

## 6. Client

A web app (works on desktop and phone browsers; the bundle format is client-agnostic, so a native
app can consume it later). Four pieces:

1. **Upload and wait.** Drag/drop or URL, progress bar from the poll. The original file stays in
   memory or an Origin Private File System handle on the device.
2. **Bundle decode.** A Web Worker streams `masks.bin` through `DecompressionStream`, parses the
   runs, and keeps per-frame mask data in typed arrays. Frames are expanded to an 8-bit id map
   (`Uint8Array` of `mask_width x mask_height`, pixel = identity id, 0 = none) lazily around the
   playhead, with a ring buffer of a few seconds so seeking stays smooth. Ids above 255 are
   remapped per frame (there are never 255 people on one frame).
3. **Live blur.** `<video>` drawn into a WebGL canvas. Each displayed frame uploads the current id
   map as a texture plus a 256-entry lookup texture "id selected or not". The fragment shader
   samples the video, samples the id map at the same UV, and if the lookup says selected, samples
   a pixelated or blurred version instead (mask dilated by a few texels in the shader). Toggling a
   label rewrites the 256-entry lookup: instant, no decode, no re-render.
4. **Selection UI.** Label toggles (`woman`, `man`, `child`) that select every identity whose
   label fraction exceeds a threshold slider; an identity gallery (thumbnail, label fractions,
   frame span) with per-id overrides; a timeline showing which ids are on screen; a "review these"
   group for conflict / flicker / ungendered ids. All derived from `manifest.json` on the device.
5. **Export.** WebCodecs: decode the original with `VideoDecoder`, run each frame through the same
   shader offscreen, encode with `VideoEncoder` (H.264, hardware where available), mux with the
   original audio track into mp4 using a small muxer library, hand the file to the user. Fallback
   for browsers without WebCodecs: `MediaRecorder` from the canvas, which yields WebM. Nothing is
   uploaded during export.

## 7. Throughput budget and levers

Server side, per job, at the measured 0.9 to 1.9 s/frame: a 30 s clip at 30 fps is 900 frames,
450 at stride 2. Levers in order: 720p cap (already the measured resolution), stride 2 default,
chunk map-reduce across workers, faster GPU tier (H100 expected 2 to 3x over the measured cards,
to be confirmed in M0). Target: a 30 s clip returns its bundle in under 3 min wall time on
Topology B, and instantly when cached.

Device side: WebGL pixelation at 720p is well under a frame's budget on any phone from the last
five years. The real constraint is bundle memory: 450 processed frames of RLE is a few MB; expanded
id maps are 230 KB each, so the ring buffer is capped at ~90 frames (~20 MB) and refilled around
the playhead.

## 8. Privacy and security

- TLS; API key per client; per-key rate and concurrency limits.
- ffprobe validation before anything else: size cap 200 MB, duration cap, container allowlist,
  fixed temp paths, ffmpeg with a timeout. Filenames never touch a shell.
- The video exists on the server only for the duration of the job. Bundles carry masks and
  thumbnails, no full frames, and expire after 24 h. `DELETE` is immediate.
- No frame content in logs; job id, counters, timings and errors only.
- Content-hash cache means re-uploading the same clip costs nothing, but the hash is salted per
  API key so one client's cache never reveals that another client uploaded the same video.

## 9. Observability

Per job in `manifest.processing` and as metrics: queue wait, cold or warm load, sec/frame, peak
VRAM, ids per prompt, bundle size, cache hit. `/healthz` on the API, worker heartbeat in Redis.
Alerts on cold start over 120 s, sec/frame over 3, bundle size over 30 MB, any failure.

## 10. Testing and acceptance

- **Golden regression.** The trial clip must yield byte-identical `tracks.jsonl` and `masks.jsonl`
  to the committed golden run (62 ids / 7131 observations for `person`) on every image build.
  Then the bundle writer must produce a byte-identical `bundle.zip` from that.
- **Chunk equivalence.** Map-reduce output equals sequential chunked output exactly.
- **Stride A/B.** Ids and escape rate at stride 1, 2, 3 on the trial clip; confirm stride 2 as the
  default and that half-resolution masks dilated by 3 px cover the full-resolution masks.
- **Bundle size.** Measure `masks.bin` on the six-clip corpus; decide RLE vs label-map video.
- **Client.** Decode a 30 s bundle on a mid-range Android phone in under 5 s; play at native fps
  with blur on; toggle a label with no visible hitch; export a 30 s clip in under 2x real time on a
  laptop. Exported mp4 opens in QuickTime, VLC, and a phone gallery with audio intact.
- **API.** Bad container, over-length, over-size, unknown job, delete, cache hit, salted hash.
- **Acceptance loop for the demo URL.** Upload a 20 s phone clip, get the bundle, see the gallery,
  toggle `woman`, scrub the timeline with blur live, export, play the export. Under 5 min end to end.

## 11. Milestones

| # | deliverable | done when |
|---|---|---|
| M0 | Benchmark on candidate GPU tiers at 720p, 4 prompts, stride 1 and 2; measure `masks.bin` size on the corpus | numbers in `service/BENCH.md`; GPU tier, stride default and mask format chosen |
| M1 | `service/worker/`: model singleton, ffmpeg ingest, fixed prompts, bundle writer, Dockerfile with baked weights, golden test through to `bundle.zip` | `docker run ... job.json` reproduces the golden bundle on a pod |
| M2 | Client: bundle decoder in a Web Worker, WebGL live blur, label and identity selection, timeline | plays the golden bundle over the trial clip with live toggles, on desktop and phone |
| M3 | Topology A: FastAPI, in-process queue, poll endpoint, client served from the pod | the acceptance loop passes on a single-pod URL |
| M4 | Client export via WebCodecs with audio, MediaRecorder fallback | exported mp4 checks in section 10 pass |
| M5 | Topology B: Serverless handler, Redis, object storage, content-hash cache, TTL sweeper, chunk map-reduce | acceptance loop against the split deployment; 30 s clip under 3 min; scales to zero |

M0 and M2 can run in parallel (M2 only needs a bundle from an existing adapter run). M1 before M3,
M3 before M5. M4 is independent of the server work once M2 exists.

## 12. Decisions needed from the owner

1. Initial label set: `woman, man, child` with `person` as control, or a different list.
2. Clip limits for v1: 30 s default, 60 s hard cap, 720p, stride 2.
3. Web client only, or a native app consuming the same bundle from day one.
4. Bundle retention: 24 h server-side, or delete on first download.
5. Object storage provider for Topology B (Cloudflare R2 vs RunPod S3-compatible).

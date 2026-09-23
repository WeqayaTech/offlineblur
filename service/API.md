# Blur service API — integration guide for the app team

Contract version 1, drafted 2026-09-23. This is the interface the app integrates against. The
service is being built to this document (see `PLAN.md`); until it is live, treat every field here
as the agreed shape and expect only additive changes before v1 ships.

## 1. What the service does and does not do

You upload a short video. The service runs SAM 3 once over it with a fixed set of labels and
returns a **bundle**: every person in the video as a stable identity, a per-frame mask for each,
and for each identity how strongly each label applied. The service **does not blur** and does not
return a video. Your app decides which identities to blur, draws the blur, and exports, all on the
device, using the bundle. That is what makes label changes instant and keeps the blurred output
off our servers.

Fixed for v1: the label set is `woman`, `man`, `child`, with `person` as the recall control. You
cannot send prompts. Query `GET /v1/labels` at startup and build your toggles from the answer so a
config change on our side does not need an app release.

## 2. Flow

```
app                                              service
---                                              -------
POST /v1/jobs (video)  ------------------------> validate, enqueue     202 {job_id}
GET  /v1/jobs/{id}     ------------------------> {status: running, progress}
   ... poll every 2 s, or receive the webhook ...
GET  /v1/jobs/{id}     ------------------------> {status: done, bundle_url}
GET  bundle_url        ------------------------> bundle.zip  (signed URL, expires)
decode bundle, show identities, blur live, export on device
DELETE /v1/jobs/{id}   (optional, when the user is finished)
```

Typical wall time for a 30 s clip: 2 to 8 minutes depending on deployment stage. Repeat uploads of
the same file return `done` immediately from cache.

## 3. Authentication

Every request carries `Authorization: Bearer <api_key>`. Keys are issued per app environment
(dev, staging, prod). A key has a concurrent-job limit (default 2) and a rate limit. Keep the key
server-side or in your app's secure storage; do not ship it in a web bundle.

## 4. Endpoints

Base URL: `https://<host>/v1` (host to be provided with the key).

### `GET /labels`

Returns the configured label set and the version of the processing config.

```json
{
  "labels": ["woman", "man", "child"],
  "control": "person",
  "config_version": "2026-09-23-5f0b3c3",
  "limits": {"max_seconds": 60, "max_bytes": 209715200, "max_long_side": 1280,
             "containers": ["mp4", "mov", "webm"]}
}
```

### `POST /jobs`

Submit a video. Two forms:

**Multipart** `Content-Type: multipart/form-data`

| field | type | required | notes |
|---|---|---|---|
| `video` | file | yes | mp4 / mov / webm, up to 200 MB, up to 60 s |
| `max_seconds` | int | no | process only the first N seconds; default 30, cap 60 |
| `stride` | int | no | process every N-th frame; default 2, allowed 1 to 3 |
| `webhook_url` | string | no | HTTPS URL we POST to when the job finishes or fails |

**JSON** `Content-Type: application/json` with `{"url": "https://...", "max_seconds": 30,
"stride": 2, "webhook_url": "..."}` for a video we can fetch. Same limits apply.

Response `202 Accepted`:

```json
{
  "job_id": "job_01J8XK4V9R2N7QW3F5T6Y8Z0AB",
  "status": "queued",
  "cached": false,
  "eta_seconds_est": 180,
  "poll_url": "/v1/jobs/job_01J8XK4V9R2N7QW3F5T6Y8Z0AB"
}
```

If the same file was processed before, `status` is `done`, `cached` is `true`, and `bundle_url`
is present in the same response.

Errors: `400 invalid_video` (not a decodable video), `413 too_large`, `422 too_long`,
`401 unauthorized`, `429 rate_limited` (with `Retry-After` seconds; also used when the queue is
full or your key is at its concurrent-job limit).

### `GET /jobs/{job_id}`

```json
{
  "job_id": "job_01J8...",
  "status": "running",
  "progress": {"frame": 210, "n_frames": 450, "sec_per_frame": 0.9, "eta_seconds": 216},
  "bundle_url": null,
  "bundle_expires_at": null,
  "error": null
}
```

`status` is one of `queued`, `running`, `done`, `failed`. When `done`, `bundle_url` is a signed
HTTPS URL valid until `bundle_expires_at` (24 h after completion). When `failed`, `error` is
`{"code": "...", "message": "..."}` with codes `timeout`, `decode_failed`, `internal`.

Poll every 2 s. `404` for an unknown or deleted job.

### `DELETE /jobs/{job_id}`

Deletes the bundle now and forgets the cache entry. `204`. Optional; bundles expire on their own.

### Webhook

If `webhook_url` was given, we POST once on `done` or `failed`:

```json
{"job_id": "job_01J8...", "status": "done", "bundle_url": "https://...", "bundle_expires_at": "..."}
```

Header `X-Blur-Signature: sha256=<hex>` is an HMAC of the raw body with your webhook secret
(issued with the key). Verify it before trusting the payload. We retry three times over 5 minutes
on non-2xx.

### `GET /healthz`

`200 {"ok": true}` when the API and the queue are reachable. No auth.

## 5. The bundle

`bundle.zip` contains `manifest.json`, `masks.bin`, and `thumbs/<id>.jpg`. Read the manifest
first; it is small and has everything except the pixels.

### `manifest.json`

```json
{
  "version": 1,
  "job_id": "job_01J8...",
  "video": {
    "source": {"width": 1920, "height": 1080},
    "width": 1280, "height": 720,
    "fps": 29.97, "n_frames": 900, "duration_s": 30.03,
    "content_sha256": "..."
  },
  "processing": {
    "stride": 2, "mask_width": 640, "mask_height": 360,
    "labels": ["woman", "man", "child"], "control": "person",
    "config_version": "2026-09-23-5f0b3c3", "seconds": 142.3
  },
  "identities": [
    {
      "id": 7,
      "first_frame": 0, "last_frame": 611, "n_obs": 298,
      "labels": {
        "woman": {"coverage": 0.91, "score": 0.83},
        "man":   {"coverage": 0.04, "score": 0.52},
        "child": {"coverage": 0.00, "score": 0.00}
      },
      "thumb": "thumbs/7.jpg",
      "box_first": [412, 88, 610, 690]
    }
  ],
  "frames": [
    {"f": 0, "ids": [7, 12], "boxes": [[412, 88, 610, 690], [900, 120, 1010, 400]]},
    {"f": 2, "ids": [7, 12], "boxes": [[414, 88, 612, 690], [898, 121, 1009, 401]]}
  ]
}
```

Coordinate systems, all pixels, origin top-left:

- `video.width x video.height` is the **processed** resolution (source downscaled so the long
  side is at most 1280). Boxes in `frames` and `box_first` are in this space. To draw on the
  source, multiply by `source.width / video.width`.
- Masks are at `mask_width x mask_height`. Scale by `video.width / mask_width` to processed space.
- Frame numbers `f` are **source** frame indices at the source fps. Only every `stride`-th frame is
  present. For any display time `t` seconds: `f = floor(t * fps)`, then look up
  `fp = floor(f / stride) * stride`. That entry's masks apply to frame `f`.

Per identity, per label:

- `coverage` is the fraction of that identity's frames the label's mask covered (0 to 1).
- `score` is the label's mean detection confidence on those frames.

### Deciding what to blur

The service ships coverages, not decisions, so the app owns the threshold. Recommended default:

```
selected(identity, label, threshold=0.5) = identity.labels[label].coverage >= threshold
verdict(identity):
  above = [l for l in labels if coverage(l) >= threshold]
  if len(above) == 1: that label
  if len(above) >  1: "conflict"
  if any 0 < coverage(l) < threshold: "flicker"
  else: "ungendered"
```

Show `conflict`, `flicker` and `ungendered` identities to the user as a review group with their
thumbnails. On the clips we have measured, about a quarter of identities land there. Do not export
from a label toggle alone without giving the user that review step; an identity that should have
been blurred and was not cannot be fixed after the video is shared.

### `masks.bin`

Gzip-compressed binary. Decompress the whole file, then parse little-endian:

```
repeat for each processed frame, in ascending f (same order as manifest.frames):
  uint32 f
  uint16 n_ids
  repeat n_ids times:
    uint16 id
    uint32 n_runs
    uint16 runs[n_runs]      alternating skip, fill, skip, fill ... over the mask grid,
                             row-major, starting with skip (COCO RLE order)
```

A run value of 65535 followed by another run of the same kind means "add these together" (runs
longer than 65534 pixels are split). Fills for different ids on the same frame do not overlap.

Decoding one frame into an id map (pixel = id, 0 = none) in JavaScript:

```js
function decodeFrame(view, offset, w, h) {
  const map = new Uint16Array(w * h);
  const f = view.getUint32(offset, true); offset += 4;
  const nIds = view.getUint16(offset, true); offset += 2;
  for (let i = 0; i < nIds; i++) {
    const id = view.getUint16(offset, true); offset += 2;
    const nRuns = view.getUint32(offset, true); offset += 4;
    let pos = 0, fill = false;
    for (let r = 0; r < nRuns; r++) {
      const len = view.getUint16(offset, true); offset += 2;
      if (fill) map.fill(id, pos, pos + len);
      pos += len; fill = !fill;
    }
  }
  return { f, map, next: offset };
}
```

In a browser, `new Response(zipEntryStream.pipeThrough(new DecompressionStream("gzip")))` gives
you the bytes without any library. On iOS or Android use the platform's gzip and the same layout.

### Blurring on the device

Per displayed frame: get the id map for `fp` as above, build a lookup of selected ids, and for each
pixel whose id is selected draw a pixelated or blurred sample instead of the source. Dilate the mask
by about 3 mask pixels (6 at processed resolution) so the half-resolution mask fully covers the
person. A fragment shader with the video and the id map as two textures and a 256-entry
"selected" lookup texture does this at native fps on a phone; toggling a label only rewrites the
lookup. Keep decoded id maps in a ring buffer around the playhead (each is `mask_width x
mask_height` bytes) rather than decoding the whole clip up front.

Export on the device by re-encoding the source with the blur applied and muxing the original audio.
Nothing is uploaded during export.

### `thumbs/<id>.jpg`

One crop per identity, taken from the frame where its score was highest, for the review UI.

## 6. Limits and retention

| limit | value |
|---|---|
| upload size | 200 MB |
| duration processed | 30 s default, 60 s cap |
| resolution | downscaled to 1280 on the long side |
| concurrent jobs per key | 2 (more on request) |
| bundle retention | 24 h after completion, then deleted |
| upload retention | deleted the moment processing ends, success or failure |

The original video never persists on the service beyond the job. Bundles contain masks and
identity thumbnails, not frames.

## 7. Versioning

`manifest.version` and the `/v1` path prefix change together. Additive fields may appear in v1
without notice; anything that changes meaning ships as `/v2` with a migration note. The
`config_version` string identifies the exact model and thresholds that produced a bundle; include
it in bug reports.

## 8. Quick test with curl

```bash
curl -s -X POST https://<host>/v1/jobs \
  -H "Authorization: Bearer $KEY" \
  -F video=@clip.mp4 -F stride=2
```

```bash
curl -s https://<host>/v1/jobs/$JOB -H "Authorization: Bearer $KEY"
```

```bash
curl -sL "$BUNDLE_URL" -o bundle.zip && unzip -l bundle.zip
```

A sample `bundle.zip` and the clip it was made from will be provided with dev keys so the app can
be built against real data before the service is live.

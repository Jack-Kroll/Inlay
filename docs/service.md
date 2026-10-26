# The job service

Inlay's pipeline runs as a service: an HTTP API takes recordings and queues
jobs, and a worker runs the pipeline on them. Today both run on this Mac. The
same code runs in the cloud by changing environment variables, not code.

```
client (the React Native app; curl for now)
   │  HTTPS
   ▼
service.api      FastAPI: uploads, jobs, results. Never loads a model.
   │                        ▲
   ▼                        │
service.jobs     the queue and job records   (SQLite file → Postgres later)
service.storage  uploads and results         (a folder    → S3/R2 later)
   ▲                        │
   │                        ▼
service.worker   claims a job, runs processing.music.transcribe.transcribe_video,
                 stores what it wrote. One job at a time, on MPS/CUDA/CPU.
```

## Run it on this Mac

```sh
uv sync --locked
uv run python -m service.api --worker          # API and one worker in one process
```

Or as two processes, which is how a deployment runs them:

```sh
uv run python -m service.api
uv run python -m service.worker
```

The worker needs the same checkpoints the CLI does (`runs/dense/fretboard-v1/best.pt`,
`runs/verifier/v11-ft1-egfx/best.pt` and its front end `runs/frontend/ft1/best.pt`,
or `runs/verifier/v8-gt/best.pt`). Jobs, uploads and results go under `var/`.

To reach it from a phone on the same Wi-Fi, bind to every interface and tell
the API the address the phone uses, because local upload URLs are built from
it:

```sh
INLAY_PUBLIC_URL=http://<mac-ip>:8000 INLAY_API_KEY=<something> \
  uv run python -m service.api --host 0.0.0.0 --worker
```

From outside the house, use [Tailscale](https://tailscale.com) and
`tailscale serve 8000`, which also gives HTTPS (iOS refuses plain HTTP by
default; Android needs `usesCleartextTraffic` for it). Do not open a port on
the router.

## The API

Interactive docs are at `/docs`. The flow:

```sh
API=http://localhost:8000
# 1. Ask where to put the recording.
curl -s -X POST $API/v1/uploads -H 'Content-Type: application/json' \
  -d '{"filename": "Blackbird.mov", "contentType": "video/quicktime"}'
# -> {"id": "<upload>", "upload": {"url": "...", "method": "PUT", "headers": {...}}}

# 2. Send it there, with the method and headers given.
curl -s -X PUT --data-binary @clips/Blackbird.mov -H 'Content-Type: video/quicktime' "<upload.url>"

# 3. Queue a job.
curl -s -X POST $API/v1/jobs -H 'Content-Type: application/json' \
  -d '{"uploadId": "<upload>", "options": {"maxFret": 22}}'
# -> 202 {"id": "<job>", "status": "queued", ...}

# 4. Poll it: queued -> running (with a stage) -> succeeded | failed.
curl -s $API/v1/jobs/<job>

# 5. Read the result: the packages/music performance document.
curl -s $API/v1/jobs/<job>/performance
```

A succeeded job also lists `artifacts`: `performance` (the document above),
`tab` (the JSONL with every diagnostic), and `overlay` (the rendered video,
only when asked for).

Job options a client may set (anything else is rejected with 422):

| field | meaning |
|---|---|
| `tuning` | open-string MIDI pitches, string 1 (high E) first |
| `maxFret` | highest fret to consider |
| `stringOrder` | `auto`, `normal` or `flipped` |
| `verifierThreshold` | 0–1; lower reports more notes |
| `audioStringWeight` | 0–1; how much audio timbre decides the string |
| `openAudioNeutral` | 0–1; how far audio is kept from arguing a note off an open string |
| `overlay` | `true` to render the overlay video |

Model paths, the device and the decoder's internals are the server's.

With `INLAY_API_KEY` set, every route except `/v1/health` needs
`Authorization: Bearer <key>` or `X-API-Key: <key>`.

## Settings

All read from the environment by `service.config.Settings`:

| variable | default | |
|---|---|---|
| `INLAY_DATA_DIR` | `var` | local storage and the SQLite file |
| `INLAY_STORAGE` | `local` | `local` or `s3` |
| `INLAY_S3_BUCKET`, `INLAY_S3_PREFIX` | | for `s3` |
| `INLAY_DATABASE` | `$INLAY_DATA_DIR/jobs.sqlite3` | the job store |
| `INLAY_PUBLIC_URL` | `http://localhost:8000` | how clients reach the API |
| `INLAY_API_KEY` | unset | require a key |
| `INLAY_DEVICE` | `mps` | `mps`, `cuda` or `cpu` |
| `INLAY_FRETBOARD_MODEL`, `INLAY_VERIFIER` | the shipped checkpoints | |
| `INLAY_WRITE_OVERLAY` | `0` | render overlays by default |
| `INLAY_MAX_UPLOAD_MB` | `1024` | |
| `INLAY_POLL_SECONDS` | `2` | idle worker poll interval |
| `INLAY_STALE_MINUTES` | `30` | a running job silent this long is requeued |

## Moving to a server

Nothing a client does changes. What changes on the server side:

- **Storage.** `INLAY_STORAGE=s3` and a bucket. The API then hands out
  presigned upload and download URLs and the bytes never pass through it.
  `S3Storage` needs `boto3` installed and is not covered by the tests, since
  that needs real credentials; R2 and other S3-compatible stores work the same
  way via boto3's endpoint settings.
- **The job store.** `SQLiteJobStore` is safe for an API and workers on one
  machine. Workers on several machines need the same five methods
  (`claim`, `progress`, `succeed`, `fail`, `requeue_stale`, plus the record
  getters) over Postgres or a managed queue. `claim` must stay atomic: one
  job, one worker.
- **Workers.** One per GPU, `INLAY_DEVICE=cuda`. The pipeline runs unchanged.
- **The process.** Run the API under a process manager behind HTTPS, with
  `INLAY_API_KEY` or real user auth in front of it.

## Limits

- A worker reloads the models for every job. That costs a few seconds per job
  and is the first thing to cache once jobs are frequent.
- One worker runs one job at a time. Jobs queue rather than run in parallel.
- Uploads through the local API are spooled to disk before they are stored,
  and are capped by `INLAY_MAX_UPLOAD_MB`.
- Nothing deletes old uploads or results yet.

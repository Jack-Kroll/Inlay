"""Inlay as a service: an HTTP API in front of a job queue a worker drains.

    uv run python -m service.api       # the API, on http://localhost:8000
    uv run python -m service.worker    # one worker: claims jobs, runs the pipeline

A client (the React Native app, eventually) never waits on a transcription.
It asks for somewhere to put a recording, uploads it there, creates a job, and
polls the job until it is done; the result is the ``packages/music``
performance document and the diagnostic files beside it.

Everything that differs between a laptop and a cloud deployment sits behind
two small interfaces, picked by environment variable (``service.config``):

- ``storage.Storage`` -- where uploads and results live. ``LocalStorage`` is a
  folder, and uploads go to the API itself; ``S3Storage`` hands out presigned
  URLs so the bytes never touch the API.
- ``jobs.JobStore`` -- the queue and job records. ``SQLiteJobStore`` is one
  file; anything with the same five methods can replace it.

The API process never imports the models. The worker loads the pipeline
(``processing.music.transcribe.transcribe_video``) and runs one job at a time.
See docs/service.md.
"""

"""The worker: claim a job, run the pipeline on it, store what it wrote.

    uv run python -m service.worker

One job at a time, on whatever device INLAY_DEVICE names. The pipeline is
``processing.music.transcribe.transcribe_video`` -- the same function the CLI
runs -- so a job's result is exactly what ``python -m processing.music.transcribe``
would have written for the same recording and options.

A worker that dies mid-job leaves the job ``running``; the next worker to
start puts anything silent for INLAY_STALE_MINUTES back on the queue. While a
job runs, a heartbeat thread keeps it from looking silent.
"""
from __future__ import annotations
import argparse
import os
import socket
import tempfile
import threading
import time
import traceback
from pathlib import Path

from .config import Settings, build
from .jobs import JobStore
from .options import options_for
from .storage import Storage

# What a finished job stores, by artifact name.
ARTIFACTS = ('performance', 'tab', 'overlay')
HEARTBEAT_SECONDS = 30.


def pipeline(video, output, options, log):
    """The real pipeline, imported only when a job runs, so importing this
    module (or the API) does not load the models."""
    from processing.music.transcribe import transcribe_video
    return transcribe_video(video, output, options, log=log)


class Worker:
    def __init__(self, settings: Settings, store: JobStore, storage: Storage,
                 transcribe=pipeline, name=None):
        self.settings = settings
        self.store = store
        self.storage = storage
        self.transcribe = transcribe
        self.name = name or f'{socket.gethostname()}:{os.getpid()}'

    def run_once(self):
        """Run the next queued job, if there is one. Returns whether there was."""
        job = self.store.claim(self.name)
        if job is None:
            return False
        stop = threading.Event()
        beat = threading.Thread(target=self._heartbeat, args=(job.id, stop), daemon=True)
        beat.start()
        try:
            self._process(job)
        except Exception as error:   # noqa: BLE001 - a failed job must not stop the worker
            traceback.print_exc()
            self.store.fail(job.id, f'{type(error).__name__}: {error}')
        finally:
            stop.set()
            beat.join(timeout=1)
        return True

    def _heartbeat(self, job_id, stop):
        while not stop.wait(HEARTBEAT_SECONDS):
            job = self.store.get_job(job_id)
            if job is not None:
                self.store.progress(job_id, job.stage)

    def _process(self, job):
        upload = self.store.get_upload(job.upload_id)
        if upload is None:
            raise LookupError(f'upload {job.upload_id} not found')
        options = options_for(job.options, self.settings)

        def log(*parts, **_):
            line = ' '.join(str(part) for part in parts)
            # Pipeline stage announcements end with an ellipsis.
            if line.endswith('...'):
                self.store.progress(job.id, line.rstrip('.').lower())

        with tempfile.TemporaryDirectory(prefix='inlay-job-') as directory:
            work = Path(directory)
            video = self.storage.fetch(upload.key, work/'input'/upload.filename)
            result = self.transcribe(video, work/'output', options, log)
            artifacts = {}
            for name, path in zip(ARTIFACTS, (result.performance, result.jsonl, result.overlay)):
                if path is None or not Path(path).is_file():
                    continue
                key = f'jobs/{job.id}/{Path(path).name}'
                self.storage.put_file(key, Path(path))
                artifacts[name] = key
        self.store.succeed(job.id, artifacts)

    def run_forever(self, stop=None):
        stop = stop or threading.Event()
        requeued = self.store.requeue_stale(self.settings.stale_minutes*60)
        if requeued:
            print(f'requeued {requeued} stale job(s)', flush=True)
        print(f'worker {self.name} waiting for jobs', flush=True)
        while not stop.is_set():
            if not self.run_once():
                stop.wait(self.settings.poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--once', action='store_true',
                        help='Run at most one job, then exit')
    args = parser.parse_args()
    settings = Settings.from_env()
    storage, store = build(settings)
    worker = Worker(settings, store, storage)
    if args.once:
        worker.run_once()
        return
    try:
        worker.run_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()

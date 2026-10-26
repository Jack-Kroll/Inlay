"""Uploads and jobs: the records, and the queue a worker claims from.

``SQLiteJobStore`` keeps both in one SQLite file. It is safe across processes
-- the API and any number of workers on one machine -- because a claim is a
single ``UPDATE ... RETURNING`` under an immediate transaction. A deployment
with workers on several machines swaps it for something with the same
methods over Postgres or a managed queue; nothing else changes.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import json
from pathlib import Path
import sqlite3
import time
import uuid
from typing import Protocol

QUEUED, RUNNING, SUCCEEDED, FAILED = 'queued', 'running', 'succeeded', 'failed'
PENDING, RECEIVED = 'pending', 'received'


def new_id():
    return uuid.uuid4().hex


@dataclass
class Upload:
    id: str
    key: str
    filename: str
    content_type: str
    status: str = PENDING
    size: int | None = None
    created: float = field(default_factory=time.time)


@dataclass
class Job:
    id: str
    upload_id: str
    options: dict
    status: str = QUEUED
    stage: str = ''
    error: str | None = None
    artifacts: dict = field(default_factory=dict)
    attempts: int = 0
    worker: str | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    heartbeat: float | None = None

    def public(self):
        """The job as the API reports it."""
        return {'id': self.id, 'uploadId': self.upload_id, 'status': self.status,
                'stage': self.stage, 'error': self.error, 'options': self.options,
                'created': self.created, 'started': self.started,
                'finished': self.finished}


class JobStore(Protocol):
    def create_upload(self, key: str, filename: str, content_type: str,
                      upload_id: str | None = None) -> Upload: ...
    def get_upload(self, upload_id: str) -> Upload | None: ...
    def mark_received(self, upload_id: str, size: int) -> None: ...
    def create_job(self, upload_id: str, options: dict) -> Job: ...
    def get_job(self, job_id: str) -> Job | None: ...
    def claim(self, worker: str) -> Job | None: ...
    def progress(self, job_id: str, stage: str) -> None: ...
    def succeed(self, job_id: str, artifacts: dict) -> None: ...
    def fail(self, job_id: str, error: str) -> None: ...
    def requeue_stale(self, older_than_seconds: float) -> int: ...


SCHEMA = '''
create table if not exists uploads (
    id text primary key, key text not null, filename text not null,
    content_type text not null, status text not null, size integer,
    created real not null);
create table if not exists jobs (
    id text primary key, upload_id text not null references uploads(id),
    options text not null, status text not null, stage text not null default '',
    error text, artifacts text not null default '{}', attempts integer not null default 0,
    worker text, created real not null, started real, finished real, heartbeat real);
create index if not exists jobs_queue on jobs(status, created);
'''


class SQLiteJobStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute('pragma journal_mode=wal')
        connection.execute('pragma foreign_keys=on')
        return _Closing(connection)

    @staticmethod
    def _upload(row):
        return None if row is None else Upload(**dict(row))

    @staticmethod
    def _job(row):
        if row is None:
            return None
        values = dict(row)
        values['options'] = json.loads(values['options'])
        values['artifacts'] = json.loads(values['artifacts'])
        return Job(**values)

    def create_upload(self, key, filename, content_type, upload_id=None):
        upload = Upload(upload_id or new_id(), key, filename, content_type)
        with self._connect() as connection:
            connection.execute(
                'insert into uploads values (?, ?, ?, ?, ?, ?, ?)',
                (upload.id, upload.key, upload.filename, upload.content_type,
                 upload.status, upload.size, upload.created))
        return upload

    def get_upload(self, upload_id):
        with self._connect() as connection:
            return self._upload(connection.execute(
                'select * from uploads where id = ?', (upload_id,)).fetchone())

    def mark_received(self, upload_id, size):
        with self._connect() as connection:
            connection.execute('update uploads set status = ?, size = ? where id = ?',
                               (RECEIVED, size, upload_id))

    def create_job(self, upload_id, options):
        job = Job(new_id(), upload_id, options)
        with self._connect() as connection:
            connection.execute(
                'insert into jobs (id, upload_id, options, status, created) '
                'values (?, ?, ?, ?, ?)',
                (job.id, upload_id, json.dumps(options), job.status, job.created))
        return job

    def get_job(self, job_id):
        with self._connect() as connection:
            return self._job(connection.execute(
                'select * from jobs where id = ?', (job_id,)).fetchone())

    def claim(self, worker):
        """The oldest queued job, now running under ``worker``; None if idle."""
        now = time.time()
        with self._connect() as connection:
            connection.execute('begin immediate')
            try:
                row = connection.execute(
                    'update jobs set status = ?, worker = ?, started = ?, heartbeat = ?, '
                    "attempts = attempts + 1, stage = 'starting' "
                    'where id = (select id from jobs where status = ? '
                    'order by created limit 1) returning *',
                    (RUNNING, worker, now, now, QUEUED)).fetchone()
                connection.execute('commit')
            except BaseException:
                connection.execute('rollback')
                raise
        return self._job(row)

    def progress(self, job_id, stage):
        with self._connect() as connection:
            connection.execute('update jobs set stage = ?, heartbeat = ? where id = ?',
                               (stage, time.time(), job_id))

    def succeed(self, job_id, artifacts):
        with self._connect() as connection:
            connection.execute(
                "update jobs set status = ?, stage = 'done', artifacts = ?, finished = ?, "
                'error = null where id = ?',
                (SUCCEEDED, json.dumps(artifacts), time.time(), job_id))

    def fail(self, job_id, error):
        with self._connect() as connection:
            connection.execute(
                'update jobs set status = ?, error = ?, finished = ? where id = ?',
                (FAILED, error, time.time(), job_id))

    def requeue_stale(self, older_than_seconds):
        """Put back running jobs whose worker went quiet, e.g. after a crash
        or the Mac sleeping. Returns how many."""
        cutoff = time.time() - older_than_seconds
        with self._connect() as connection:
            return connection.execute(
                "update jobs set status = ?, worker = null, stage = 'requeued' "
                'where status = ? and coalesce(heartbeat, started, created) < ?',
                (QUEUED, RUNNING, cutoff)).rowcount


class _Closing:
    """sqlite3's connection context manager commits but does not close."""

    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self.connection

    def __exit__(self, *exc):
        self.connection.close()
        return False

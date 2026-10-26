"""Service settings, read from the environment.

Every switch between running on this Mac and running in the cloud is one of
these variables; the code paths do not otherwise know where they are.

    INLAY_DATA_DIR        local storage and the SQLite file (default: var)
    INLAY_STORAGE         local | s3 (default: local)
    INLAY_S3_BUCKET       bucket for s3 storage
    INLAY_S3_PREFIX       key prefix inside it (default: none)
    INLAY_DATABASE        SQLite path for the job store (default: $INLAY_DATA_DIR/jobs.sqlite3)
    INLAY_PUBLIC_URL      how clients reach the API, for local upload URLs
                          (default: http://localhost:8000)
    INLAY_API_KEY         when set, every request but /v1/health must carry it
    INLAY_DEVICE          mps | cuda | cpu for the worker (default: mps)
    INLAY_FRETBOARD_MODEL fretboard checkpoint (default: the shipped one)
    INLAY_VERIFIER        verifier checkpoint (default: the shipped one)
    INLAY_WRITE_OVERLAY   1 to render the overlay video for every job (default: 0)
    INLAY_MAX_UPLOAD_MB   largest accepted upload (default: 1024)
    INLAY_POLL_SECONDS    how often an idle worker looks for work (default: 2)
    INLAY_STALE_MINUTES   a running job silent this long is requeued (default: 30)
"""
from __future__ import annotations
from dataclasses import dataclass
import os
from pathlib import Path

from processing.vision import DEFAULT_MODEL


def _flag(value):
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path('var')
    storage: str = 'local'
    s3_bucket: str | None = None
    s3_prefix: str = ''
    database: Path | None = None
    public_url: str = 'http://localhost:8000'
    api_key: str | None = None
    device: str = 'mps'
    model: Path = DEFAULT_MODEL
    verifier: Path | None = None
    write_overlay: bool = False
    max_upload_bytes: int = 1024*1024*1024
    poll_seconds: float = 2.
    stale_minutes: float = 30.

    @property
    def database_path(self):
        return self.database or self.data_dir/'jobs.sqlite3'

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env

        def get(name, default=''):
            return env.get(name) or default

        data_dir = Path(get('INLAY_DATA_DIR', 'var'))
        verifier = get('INLAY_VERIFIER')
        database = get('INLAY_DATABASE')
        return cls(
            data_dir=data_dir,
            storage=get('INLAY_STORAGE', 'local'),
            s3_bucket=get('INLAY_S3_BUCKET') or None,
            s3_prefix=get('INLAY_S3_PREFIX', ''),
            database=Path(database) if database else None,
            public_url=get('INLAY_PUBLIC_URL', 'http://localhost:8000').rstrip('/'),
            api_key=get('INLAY_API_KEY') or None,
            device=get('INLAY_DEVICE', 'mps'),
            model=Path(get('INLAY_FRETBOARD_MODEL', str(DEFAULT_MODEL))),
            verifier=Path(verifier) if verifier else None,
            write_overlay=_flag(get('INLAY_WRITE_OVERLAY', '0')),
            max_upload_bytes=int(float(get('INLAY_MAX_UPLOAD_MB', '1024'))*1024*1024),
            poll_seconds=float(get('INLAY_POLL_SECONDS', '2')),
            stale_minutes=float(get('INLAY_STALE_MINUTES', '30')))


def build(settings):
    """The storage and job store these settings describe."""
    from .jobs import SQLiteJobStore
    from .storage import LocalStorage, S3Storage
    if settings.storage == 'local':
        storage = LocalStorage(settings.data_dir/'storage', settings.public_url)
    elif settings.storage == 's3':
        if not settings.s3_bucket:
            raise ValueError('INLAY_STORAGE=s3 needs INLAY_S3_BUCKET')
        storage = S3Storage(settings.s3_bucket, settings.s3_prefix)
    else:
        raise ValueError(f'Unknown INLAY_STORAGE {settings.storage!r}; use local or s3')
    return storage, SQLiteJobStore(settings.database_path)

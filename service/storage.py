"""Where uploads and results live.

The client protocol is the same whatever the backend: ask the API for an
upload target, send the file there with the method and headers it names, and
later fetch results from the URLs a finished job lists. Locally the target is
the API itself; on S3 it is a presigned URL and the bytes never pass through
the API. Swapping one for the other changes nothing a client does.

Keys are plain relative paths: ``uploads/<id>/<filename>`` and
``jobs/<id>/<artifact>``.
"""
from __future__ import annotations
import shutil
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable, Protocol


def _safe(key):
    """Reject a key that could escape the storage root."""
    path = PurePosixPath(key)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError(f'Invalid storage key {key!r}')
    return str(path)


class Storage(Protocol):
    def upload_target(self, key: str, upload_id: str, content_type: str) -> dict:
        """Where a client sends a new upload: {'url', 'method', 'headers'}."""
        ...

    def write(self, key: str, chunks: Iterable[bytes], limit: int) -> int:
        """Store a stream under ``key``; returns its size. Only backends whose
        uploads come through the API need this to work."""
        ...

    def put_file(self, key: str, source: Path) -> None: ...

    def fetch(self, key: str, destination: Path) -> Path: ...

    def exists(self, key: str) -> bool: ...

    def size(self, key: str) -> int | None: ...

    def download_url(self, key: str) -> str | None:
        """A URL a client can fetch directly, or None when the API serves it."""
        ...

    def local_path(self, key: str) -> Path | None:
        """The file on this machine, when there is one."""
        ...


class LocalStorage:
    """A directory. Uploads come through the API's own upload route."""

    def __init__(self, root, public_url='http://localhost:8000'):
        self.root = Path(root)
        self.public_url = public_url.rstrip('/')

    def path(self, key):
        return self.root/_safe(key)

    def upload_target(self, key, upload_id, content_type):
        return {'url': f'{self.public_url}/v1/uploads/{upload_id}/content', 'method': 'PUT',
                'headers': {'Content-Type': content_type}}

    def write(self, key, chunks, limit):
        destination = self.path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + '.part')
        size = 0
        try:
            with partial.open('wb') as handle:
                for chunk in chunks:
                    size += len(chunk)
                    if size > limit:
                        raise ValueError(f'Upload exceeds {limit} bytes')
                    handle.write(chunk)
            partial.replace(destination)
        finally:
            partial.unlink(missing_ok=True)
        return size

    def put_file(self, key, source):
        destination = self.path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    def fetch(self, key, destination):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.path(key), destination)
        return destination

    def exists(self, key):
        return self.path(key).is_file()

    def size(self, key):
        path = self.path(key)
        return path.stat().st_size if path.is_file() else None

    def download_url(self, key):
        return None

    def local_path(self, key):
        path = self.path(key)
        return path if path.is_file() else None


class S3Storage:
    """An S3 (or S3-compatible, e.g. R2) bucket, through presigned URLs.

    Not exercised by the test suite: it needs credentials and a bucket.
    ``boto3`` is imported on first use, so it is only a dependency of a
    deployment that sets INLAY_STORAGE=s3. Credentials come from boto3's usual
    chain (environment, profile, instance role)."""

    expires = 3600

    def __init__(self, bucket, prefix=''):
        self.bucket = bucket
        self.prefix = prefix.strip('/')
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import boto3  # pyright: ignore[reportMissingImports]  # Only an s3 deployment installs it.
            self._client = boto3.client('s3')
        return self._client

    def _key(self, key):
        key = _safe(key)
        return f'{self.prefix}/{key}' if self.prefix else key

    def upload_target(self, key, upload_id, content_type):
        url = self.client.generate_presigned_url(
            'put_object', Params={'Bucket': self.bucket, 'Key': self._key(key),
                                  'ContentType': content_type},
            ExpiresIn=self.expires)
        return {'url': url, 'method': 'PUT', 'headers': {'Content-Type': content_type}}

    def write(self, key, chunks, limit):
        raise NotImplementedError('S3 uploads go straight to the presigned URL')

    def put_file(self, key, source):
        self.client.upload_file(str(source), self.bucket, self._key(key))

    def fetch(self, key, destination):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, self._key(key), str(destination))
        return destination

    def size(self, key):
        try:
            return int(self.client.head_object(Bucket=self.bucket,
                                               Key=self._key(key))['ContentLength'])
        except self.client.exceptions.ClientError:
            return None

    def exists(self, key):
        return self.size(key) is not None

    def download_url(self, key):
        return self.client.generate_presigned_url(
            'get_object', Params={'Bucket': self.bucket, 'Key': self._key(key)},
            ExpiresIn=self.expires)

    def local_path(self, key):
        return None


def open_local(storage: Storage, key: str) -> BinaryIO | None:
    path = storage.local_path(key)
    return path.open('rb') if path is not None else None

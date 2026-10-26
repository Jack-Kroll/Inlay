"""The HTTP API: uploads in, jobs queued, results out.

    uv run python -m service.api                  # http://127.0.0.1:8000
    uv run python -m service.api --host 0.0.0.0   # reachable from a phone on the LAN
    uv run python -m service.api --worker         # and run a worker in-process

The flow a client follows, the same against this Mac or a cloud deployment:

    POST /v1/uploads            {filename, contentType} -> {id, upload: {url, method, headers}}
    <method> <upload.url>       the recording's bytes
    POST /v1/jobs               {uploadId, options} -> 202 {id, status: queued}
    GET  /v1/jobs/{id}          poll until status is succeeded or failed
    GET  /v1/jobs/{id}/performance   the packages/music performance document

When INLAY_API_KEY is set every route but /v1/health needs it, as
``Authorization: Bearer <key>`` or ``X-API-Key: <key>``. Interactive docs are
at /docs.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import tempfile
import threading

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field

from .config import Settings, build
from .jobs import RECEIVED, SUCCEEDED, JobStore, new_id
from .options import options_for
from .storage import Storage, LocalStorage

SPOOL_CHUNK = 1 << 20


class UploadRequest(BaseModel):
    filename: str = 'recording.mov'
    contentType: str = 'video/quicktime'


class JobRequest(BaseModel):
    uploadId: str
    options: dict = Field(default_factory=dict)


def _filename(name):
    """A storage-safe version of a client's filename, keeping its extension:
    the pipeline names its outputs after it."""
    name = re.sub(r'[^A-Za-z0-9._ -]+', '', name.rsplit('/', 1)[-1]).strip(' .')
    return name[:120] or 'recording'


def create_app(settings: Settings | None = None, store: JobStore | None = None,
               storage: Storage | None = None):
    settings = settings or Settings.from_env()
    if store is None or storage is None:
        built_storage, built_store = build(settings)
        storage = storage or built_storage
        store = store or built_store
    assert store is not None and storage is not None
    app = FastAPI(title='Inlay', version='1',
                  description='Guitar transcription jobs: upload a recording, get tab back.')

    def authorize(request: Request):
        if not settings.api_key:
            return
        bearer = request.headers.get('authorization', '')
        supplied = (bearer[7:] if bearer.lower().startswith('bearer ')
                    else request.headers.get('x-api-key'))
        if supplied != settings.api_key:
            raise HTTPException(401, 'Missing or wrong API key')

    def job_or_404(job_id):
        job = store.get_job(job_id)
        if job is None:
            raise HTTPException(404, f'No job {job_id}')
        return job

    def artifact_url(job_id, name, key):
        return (storage.download_url(key)
                or f'{settings.public_url}/v1/jobs/{job_id}/artifacts/{name}')

    @app.get('/v1/health')
    def health():
        return {'status': 'ok', 'storage': settings.storage}

    @app.post('/v1/uploads', status_code=201, dependencies=[Depends(authorize)])
    def create_upload(body: UploadRequest):
        upload_id = new_id()
        filename = _filename(body.filename)
        key = f'uploads/{upload_id}/{filename}'
        store.create_upload(key, filename, body.contentType, upload_id)
        return {'id': upload_id,
                'upload': storage.upload_target(key, upload_id, body.contentType)}

    @app.put('/v1/uploads/{upload_id}/content', dependencies=[Depends(authorize)])
    async def put_content(upload_id: str, request: Request):
        upload = store.get_upload(upload_id)
        if upload is None:
            raise HTTPException(404, f'No upload {upload_id}')
        if not isinstance(storage, LocalStorage):
            raise HTTPException(405, 'This deployment takes uploads at the URL '
                                     'POST /v1/uploads returned')
        limit = settings.max_upload_bytes
        # Spooled first so the storage write stays synchronous and the size
        # limit is enforced before anything is kept.
        with tempfile.SpooledTemporaryFile(max_size=64*SPOOL_CHUNK) as spool:
            received = 0
            async for chunk in request.stream():
                received += len(chunk)
                if received > limit:
                    raise HTTPException(413, f'Upload exceeds {limit} bytes')
                spool.write(chunk)
            if not received:
                raise HTTPException(400, 'Empty upload')
            spool.seek(0)
            size = storage.write(upload.key, iter(lambda: spool.read(SPOOL_CHUNK), b''), limit)
        store.mark_received(upload_id, size)
        return {'id': upload_id, 'size': size}

    @app.post('/v1/jobs', status_code=202, dependencies=[Depends(authorize)])
    def create_job(body: JobRequest):
        upload = store.get_upload(body.uploadId)
        if upload is None:
            raise HTTPException(404, f'No upload {body.uploadId}')
        if upload.status != RECEIVED:
            # Presigned uploads never pass through the API; look for the object.
            size = storage.size(upload.key)
            if size is None:
                raise HTTPException(409, f'Upload {upload.id} has no content yet')
            store.mark_received(upload.id, size)
        try:
            options_for(body.options, settings)
        except ValueError as error:
            raise HTTPException(422, str(error))
        job = store.create_job(upload.id, body.options)
        return job.public()

    @app.get('/v1/jobs/{job_id}', dependencies=[Depends(authorize)])
    def get_job(job_id: str):
        job = job_or_404(job_id)
        report = job.public()
        if job.status == SUCCEEDED:
            report['artifacts'] = {name: artifact_url(job.id, name, key)
                                   for name, key in job.artifacts.items()}
        return report

    @app.get('/v1/jobs/{job_id}/performance', dependencies=[Depends(authorize)])
    def get_performance(job_id: str):
        job = job_or_404(job_id)
        if job.status != SUCCEEDED or 'performance' not in job.artifacts:
            raise HTTPException(409, f'Job {job_id} is {job.status}')
        key = job.artifacts['performance']
        path = storage.local_path(key)
        if path is not None:
            return json.loads(path.read_text())
        with tempfile.TemporaryDirectory() as directory:
            fetched = storage.fetch(key, Path(directory)/'performance.json')
            return json.loads(fetched.read_text())

    @app.get('/v1/jobs/{job_id}/artifacts/{name}', dependencies=[Depends(authorize)])
    def get_artifact(job_id: str, name: str):
        job = job_or_404(job_id)
        key = job.artifacts.get(name)
        if job.status != SUCCEEDED or key is None:
            raise HTTPException(404, f'Job {job_id} has no {name}')
        url = storage.download_url(key)
        if url:
            return RedirectResponse(url)
        path = storage.local_path(key)
        if path is None:
            raise HTTPException(404, f'{name} is missing from storage')
        return FileResponse(path, filename=path.name)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--worker', action='store_true',
                        help='Also run a worker thread in this process (development)')
    args = parser.parse_args()
    import uvicorn
    settings = Settings.from_env()
    storage, store = build(settings)
    if args.worker:
        from .worker import Worker
        threading.Thread(target=Worker(settings, store, storage).run_forever,
                         daemon=True).start()
    uvicorn.run(create_app(settings, store, storage), host=args.host, port=args.port)


if __name__ == '__main__':
    main()

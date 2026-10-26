"""The job service end to end, with the pipeline stubbed out: no models needed."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from processing.music.transcribe import Transcription, performance
from service.api import create_app
from service.config import Settings, build
from service.jobs import FAILED, QUEUED, RUNNING, SUCCEEDED, SQLiteJobStore
from service.options import options_for
from service.worker import Worker


class Note:
    def __init__(self, pitch, start, end, string=None, fret=None, confidence=.9):
        self.pitch, self.start, self.end = pitch, start, end
        self.string, self.fret, self.confidence = string, fret, confidence


def fake_pipeline(calls):
    def transcribe(video, output, options, log):
        calls.append((Path(video), Path(video).read_bytes(), options))
        log('Detecting pitch...', flush=True)
        output.mkdir(parents=True, exist_ok=True)
        notes = {0: Note(64, .5, 1., 1, 0), 1: Note(67, 1., 1.5)}
        document = output/f'{video.stem}.performance.json'
        document.write_text(json.dumps(performance(notes, options.tuning)))
        jsonl = output/f'{video.stem}.tab.jsonl'
        jsonl.write_text('{"kind": "summary"}\n')
        return Transcription(jsonl, document, None, {}, notes)
    return transcribe


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.settings = Settings(data_dir=root, public_url='http://testserver',
                                 model=root/'fretboard.pt', poll_seconds=.01)
        self.storage, self.store = build(self.settings)
        self.client = TestClient(create_app(self.settings, self.store, self.storage))
        self.calls = []
        self.worker = Worker(self.settings, self.store, self.storage,
                             fake_pipeline(self.calls), name='test')

    def tearDown(self):
        self.directory.cleanup()

    def upload(self, content=b'not really a video', filename='My Take.mov'):
        response = self.client.post('/v1/uploads', json={'filename': filename,
                                                          'contentType': 'video/quicktime'})
        self.assertEqual(response.status_code, 201)
        body = response.json()
        target = body['upload']
        self.assertEqual(target['method'], 'PUT')
        self.assertTrue(target['url'].startswith('http://testserver/v1/uploads/'))
        path = target['url'][len('http://testserver'):]
        put = self.client.put(path, content=content, headers=target['headers'])
        self.assertEqual(put.status_code, 200, put.text)
        self.assertEqual(put.json()['size'], len(content))
        return body['id']

    def test_upload_job_and_result_round_trip(self):
        upload = self.upload()
        created = self.client.post('/v1/jobs', json={'uploadId': upload,
                                                     'options': {'maxFret': 20}})
        self.assertEqual(created.status_code, 202, created.text)
        job = created.json()
        self.assertEqual(job['status'], QUEUED)
        self.assertEqual(self.client.get(f'/v1/jobs/{job["id"]}/performance').status_code, 409)

        self.assertTrue(self.worker.run_once())
        self.assertFalse(self.worker.run_once())
        video, content, options = self.calls[0]
        self.assertEqual(content, b'not really a video')
        self.assertEqual(video.name, 'My Take.mov')
        self.assertEqual(options.max_fret, 20)
        self.assertEqual(options.model, self.settings.model)
        self.assertFalse(options.write_video)

        report = self.client.get(f'/v1/jobs/{job["id"]}').json()
        self.assertEqual(report['status'], SUCCEEDED)
        self.assertEqual(set(report['artifacts']), {'performance', 'tab'})
        document = self.client.get(f'/v1/jobs/{job["id"]}/performance').json()
        self.assertEqual(document['version'], 1)
        self.assertEqual(document['notes'][0], {'pitch': 64, 'startSeconds': .5,
                                                'durationSeconds': .5, 'confidence': .9,
                                                'string': 1, 'fret': 0})
        self.assertNotIn('string', document['notes'][1])
        tab = self.client.get(report['artifacts']['tab'][len('http://testserver'):])
        self.assertEqual(tab.status_code, 200)
        self.assertEqual(tab.text, '{"kind": "summary"}\n')

    def test_job_needs_uploaded_content(self):
        response = self.client.post('/v1/uploads', json={})
        upload = response.json()['id']
        self.assertEqual(self.client.post('/v1/jobs', json={'uploadId': upload}).status_code, 409)
        self.assertEqual(self.client.post('/v1/jobs', json={'uploadId': 'nope'}).status_code, 404)

    def test_bad_options_are_rejected_before_queueing(self):
        upload = self.upload()
        for options in ({'device': 'cuda'}, {'maxFret': 0}, {'maxFret': True},
                        {'verifierThreshold': 2}, {'tuning': [64, 'x']},
                        {'stringOrder': 'sideways'}):
            with self.subTest(options=options):
                response = self.client.post('/v1/jobs', json={'uploadId': upload,
                                                              'options': options})
                self.assertEqual(response.status_code, 422, response.text)
        self.assertIsNone(self.store.claim('test'))

    def test_failed_pipeline_marks_the_job_failed_and_keeps_working(self):
        def broken(video, output, options, log):
            raise RuntimeError('no frames')
        worker = Worker(self.settings, self.store, self.storage, broken, name='test')
        job = self.client.post('/v1/jobs', json={'uploadId': self.upload()}).json()
        with patch('traceback.print_exc'):
            self.assertTrue(worker.run_once())
        report = self.client.get(f'/v1/jobs/{job["id"]}').json()
        self.assertEqual(report['status'], FAILED)
        self.assertEqual(report['error'], 'RuntimeError: no frames')
        self.assertNotIn('artifacts', report)

    def test_api_key_guards_everything_but_health(self):
        settings = replace(self.settings, api_key='secret')
        client = TestClient(create_app(settings, self.store, self.storage))
        self.assertEqual(client.get('/v1/health').status_code, 200)
        self.assertEqual(client.post('/v1/uploads', json={}).status_code, 401)
        self.assertEqual(client.post('/v1/uploads', json={},
                                     headers={'X-API-Key': 'wrong'}).status_code, 401)
        self.assertEqual(client.post('/v1/uploads', json={},
                                     headers={'Authorization': 'Bearer secret'}).status_code, 201)

    def test_oversized_and_empty_uploads_are_refused(self):
        settings = replace(self.settings, max_upload_bytes=10)
        client = TestClient(create_app(settings, self.store, self.storage))
        upload = client.post('/v1/uploads', json={}).json()['id']
        self.assertEqual(client.put(f'/v1/uploads/{upload}/content',
                                    content=b'x'*11).status_code, 413)
        self.assertEqual(client.put(f'/v1/uploads/{upload}/content',
                                    content=b'').status_code, 400)
        stored = self.store.get_upload(upload)
        assert stored is not None
        self.assertEqual(stored.status, 'pending')

    def test_filenames_cannot_escape_storage(self):
        response = self.client.post('/v1/uploads', json={'filename': '../../etc/passwd'})
        upload = self.store.get_upload(response.json()['id'])
        assert upload is not None
        self.assertEqual(upload.filename, 'passwd')
        self.assertTrue(upload.key.startswith('uploads/'))


class JobStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = SQLiteJobStore(Path(self.directory.name)/'jobs.sqlite3')
        self.upload = self.store.create_upload('uploads/a/x.mov', 'x.mov', 'video/quicktime')

    def tearDown(self):
        self.directory.cleanup()

    def test_claims_are_oldest_first_and_never_shared(self):
        jobs = [self.store.create_job(self.upload.id, {}) for _ in range(20)]
        claimed, lock = [], threading.Lock()

        def drain():
            while True:
                job = self.store.claim(threading.current_thread().name)
                if job is None:
                    return
                with lock:
                    claimed.append(job.id)
        threads = [threading.Thread(target=drain, name=f'w{i}') for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(sorted(claimed), sorted(job.id for job in jobs))
        self.assertEqual(len(set(claimed)), len(claimed))

    def test_stale_running_jobs_are_requeued(self):
        job = self.store.create_job(self.upload.id, {'maxFret': 12})
        claimed = self.store.claim('crashed')
        assert claimed is not None
        self.assertEqual((claimed.id, claimed.status, claimed.attempts), (job.id, RUNNING, 1))
        self.assertEqual(self.store.requeue_stale(3600), 0)
        self.assertEqual(self.store.requeue_stale(-1), 1)
        again = self.store.claim('fresh')
        assert again is not None
        self.assertEqual((again.id, again.attempts, again.options), (job.id, 2, {'maxFret': 12}))


class OptionTests(unittest.TestCase):
    def test_client_fields_map_onto_server_defaults(self):
        settings = Settings(model=Path('m.pt'), device='cpu', write_overlay=True)
        options = options_for({'tuning': [62, 57, 53, 48, 43, 38], 'overlay': False,
                               'audioStringWeight': 0}, settings)
        self.assertEqual(options.tuning, (62, 57, 53, 48, 43, 38))
        self.assertEqual((options.model, options.device), (Path('m.pt'), 'cpu'))
        self.assertFalse(options.write_video)
        self.assertEqual(options.audio_string_weight, 0)

    def test_settings_read_the_environment(self):
        settings = Settings.from_env({'INLAY_DATA_DIR': '/tmp/inlay', 'INLAY_STORAGE': 's3',
                                      'INLAY_S3_BUCKET': 'b', 'INLAY_WRITE_OVERLAY': 'true',
                                      'INLAY_MAX_UPLOAD_MB': '2', 'INLAY_DEVICE': 'cuda',
                                      'INLAY_PUBLIC_URL': 'https://inlay.example/'})
        self.assertEqual(settings.database_path, Path('/tmp/inlay/jobs.sqlite3'))
        self.assertEqual((settings.storage, settings.s3_bucket, settings.device),
                         ('s3', 'b', 'cuda'))
        self.assertTrue(settings.write_overlay)
        self.assertEqual(settings.max_upload_bytes, 2*1024*1024)
        self.assertEqual(settings.public_url, 'https://inlay.example')
        with self.assertRaises(ValueError):
            build(replace(settings, s3_bucket=None))


if __name__ == '__main__':
    unittest.main()

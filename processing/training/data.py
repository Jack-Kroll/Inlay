"""Unified, partially supervised fretboard data. Coordinates are normalized XY."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

HEADS = ('neck', 'fret', 'nut')
MEAN = np.array([.485, .456, .406], np.float32)
STD = np.array([.229, .224, .225], np.float32)


def read_manifest(path):
    path = Path(path).resolve()
    doc = json.loads(path.read_text())
    if doc.get('version') != 1 or doc.get('heads') != list(HEADS):
        raise ValueError('Expected version 1 manifest with neck/fret/nut heads')
    records = doc['records']
    groups, hashes = {}, {}
    for r in records:
        if r['split'] not in ('train', 'val', 'test'):
            raise ValueError('Invalid split')
        if not r.get('group') or not set(r['supervised']).issubset(HEADS):
            raise ValueError('Every record needs a group and valid supervised heads')
        if not r['supervised']:
            raise ValueError('Record has no supervision')
        for key, registry in ((r['group'], groups), (r.get('sha256'), hashes)):
            if key and registry.setdefault(key, r['split']) != r['split']:
                raise ValueError(f'Cross-split leakage: {key}')
        r['image'] = str((path.parent / r['image']).resolve())
        if not Path(r['image']).is_file():
            raise FileNotFoundError(r['image'])
        if r.get('sha256') and hashlib.sha256(Path(r['image']).read_bytes()).hexdigest() != r['sha256']:
            raise ValueError(f"Image changed since dataset preparation: {r['image']}")
        if not r.get('source'):
            raise ValueError('Every record needs a source name')
        r.setdefault('original_group', r['group'])
        for label in r['annotations']:
            pts = np.asarray(label['points'], float)
            if label['head'] not in r['supervised'] or label['kind'] not in ('polygon', 'line'):
                raise ValueError('Invalid annotation or unsupervised target')
            if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < (3 if label['kind'] == 'polygon' else 2):
                raise ValueError('Malformed points')
            if not np.isfinite(pts).all() or (pts < 0).any() or (pts > 1).any():
                raise ValueError('Coordinates must be finite and normalized to [0, 1]')
    return doc


def letterbox(image, size):
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), 114, np.uint8)
    out[top:top+nh, left:left+nw] = cv2.resize(image, (nw, nh))
    return out, (left, top, nw, nh)


def image_tensor(image):
    rgb = image[..., ::-1].astype(np.float32) / 255
    return torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1).copy())


def render_targets(record, shape, size):
    h, w = shape[:2]
    _, (left, top, nw, nh) = letterbox(np.zeros((h, w, 3), np.uint8), size)
    target = np.zeros((size, size, 3), np.float32)
    for label in record['annotations']:
        ch = HEADS.index(label['head'])
        pts = np.rint(np.array(label['points']) * [nw, nh] + [left, top]).astype(np.int32)
        # Normalized boundary points (1.0) belong to the last image pixel,
        # otherwise edge wires disappear into padding or outside the canvas.
        pts = np.clip(pts, [left, top], [left+nw-1, top+nh-1]).astype(np.int32)
        plane = np.zeros((size, size), np.uint8)
        if label['kind'] == 'polygon':
            cv2.fillPoly(plane, [pts], 1)
        else:
            # Finite-width wire supervision, not decorative inlay targets.
            cv2.polylines(plane, [pts], False, 1, max(1, round(size / 768)))
        target[..., ch] = np.maximum(target[..., ch], plane)
    valid = np.zeros_like(target)
    for head in record['supervised']:
        valid[top:top+nh, left:left+nw, HEADS.index(head)] = 1
    return target, valid


def augment(image, target, valid, rng):
    s = image.shape[0]
    matrix = cv2.getRotationMatrix2D((s/2, s/2), rng.uniform(-35, 35), rng.uniform(.8, 1.2))
    matrix[:, 2] += rng.uniform(-.12, .12, 2) * s
    matrix = np.vstack([matrix, np.array([rng.uniform(-.00015, .00015), rng.uniform(-.00015, .00015), 1])])
    image = cv2.warpPerspective(image, matrix, (s, s), borderValue=(114, 114, 114))
    target = cv2.warpPerspective(target, matrix, (s, s), flags=cv2.INTER_NEAREST)
    valid = cv2.warpPerspective(valid, matrix, (s, s), flags=cv2.INTER_NEAREST)
    if rng.random() < .5:
        image, target, valid = image[:, ::-1].copy(), target[:, ::-1].copy(), valid[:, ::-1].copy()
    image = np.clip(image.astype(float) * rng.uniform(.65, 1.35) + rng.normal(0, 4, image.shape), 0, 255).astype(np.uint8)
    if rng.random() < .25:
        image = cv2.GaussianBlur(image, (3, 3), rng.uniform(.3, 1.5))
    if rng.random() < .5:
        # Localized occlusion: preserve known amodal targets underneath the cover.
        # This is synthetic robustness training, not a claim of real-hand accuracy.
        ys, xs = np.where(target[..., 0] > .5)
        if len(xs):
            i = rng.integers(len(xs)); cx, cy = xs[i], ys[i]
            rw, rh = rng.uniform(.035, .15, 2) * s
            x0, x1 = max(0, int(cx-rw)), min(s, int(cx+rw))
            y0, y1 = max(0, int(cy-rh)), min(s, int(cy+rh))
            image[y0:y1, x0:x1] = rng.integers(20, 235, 3, dtype=np.uint8)
    return image, target, valid


class FretboardDataset(Dataset):
    def __init__(self, manifest, split, size=768, training=False, seed=42):
        self.records = [r for r in read_manifest(manifest)['records'] if r['split'] == split]
        if not self.records:
            raise ValueError(f'No records for {split}')
        self.size, self.training, self.seed, self.epoch = size, training, seed, 0

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        image = cv2.imread(r['image'])
        if image is None:
            raise ValueError(f"Cannot decode {r['image']}")
        target, valid = render_targets(r, image.shape, self.size)
        image, _ = letterbox(image, self.size)
        if self.training:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
            image, target, valid = augment(image, target, valid, rng)
        return image_tensor(image), torch.from_numpy(target.transpose(2, 0, 1).copy()), torch.from_numpy(valid.transpose(2, 0, 1).copy())


def manifest_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

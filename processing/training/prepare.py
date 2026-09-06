"""Audit and merge local YOLO OBB / COCO polygon exports into a dense manifest."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import yaml
from .data import HEADS, read_manifest


def source_key(path):
    return Path(path).stem.split('.rf.')[0]


def phash(image):
    gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (32, 32)).astype(np.float32)
    d = cv2.dct(gray)[:8, :8].flatten()[1:]
    return sum(int(v > np.median(d)) << i for i, v in enumerate(d))


def wire_centerline(points):
    # OBB edges are ordered; the midpoints of the short opposite edges define the wire.
    p = np.asarray(points, float)
    lengths = np.linalg.norm(np.roll(p, -1, axis=0)-p, axis=1)
    i = int(np.argmin(lengths))
    return [(p[i]+p[(i+1)%4])/2, (p[(i+2)%4]+p[(i+3)%4])/2]


def import_source(spec):
    root = Path(spec['root']).resolve()
    mapping = spec['mapping']
    supervised = spec['supervised']
    if not set(supervised).issubset(HEADS) or not supervised:
        raise ValueError('Declare which heads are exhaustively labeled in this source')
    if not spec.get('attribution') or not spec.get('license'):
        raise ValueError('Every source requires attribution and license text')
    records = []
    for folder, split in [('train', 'train'), ('valid', 'val'), ('val', 'val'), ('test', 'test')]:
        directory = root / folder
        if not directory.exists():
            continue
        annotations = {}
        if spec['format'] == 'obb':
            config = yaml.safe_load((root/'data.yaml').read_text())
            names = config['names']
            names = {int(k): v for k,v in names.items()} if isinstance(names, dict) else dict(enumerate(names))
            for image in sorted((directory/'images').glob('*')):
                if image.suffix.lower() not in ('.png','.jpg','.jpeg','.webp'):
                    continue
                label_path = directory/'labels'/f'{image.stem}.txt'
                if not label_path.is_file():
                    raise ValueError(f'Missing label file (use an empty file for reviewed negatives): {label_path}')
                labels = []
                for row in label_path.read_text().splitlines():
                    values = row.split()
                    if not values:
                        continue
                    if len(values) != 9:
                        raise ValueError(f'Expected OBB row: {label_path}')
                    name = names[int(values[0])]
                    if name not in mapping:
                        continue
                    head = mapping[name]
                    p = np.array(values[1:], float).reshape(4,2)
                    if not np.isfinite(p).all() or (p < -.25).any() or (p > 1.25).any():
                        raise ValueError(f'Invalid normalized OBB: {label_path}')
                    clipped = bool((p < 0).any() or (p > 1).any())
                    kind = 'polygon' if head == 'neck' else 'line'
                    if kind == 'line':
                        p = np.array(wire_centerline(p))
                    if kind == 'polygon':
                        _, intersection = cv2.intersectConvexConvex(p.astype(np.float32),np.array([[0,0],[1,0],[1,1],[0,1]],np.float32))
                        if intersection is None: continue
                        p = intersection.reshape(-1,2).clip(0,1)
                    else:
                        scale = 1000000
                        ok, a, b = cv2.clipLine((0,0,scale+1,scale+1),tuple(np.rint(p[0]*scale).astype(int)),tuple(np.rint(p[1]*scale).astype(int)))
                        if not ok: continue
                        p = np.array([a,b],float)/scale
                    labels.append({'head':head, 'kind':kind, 'points':p.tolist(), 'clipped':clipped})
                annotations[image] = labels
        elif spec['format'] == 'coco':
            doc = json.loads((directory/spec.get('annotation_file', '_annotations.coco.json')).read_text())
            names = {c['id']:c['name'] for c in doc['categories']}
            by_image = defaultdict(list)
            for a in doc['annotations']:
                if names[a['category_id']] in mapping:
                    by_image[a['image_id']].append(a)
            for im in doc['images']:
                labels = []
                for a in by_image[im['id']]:
                    head = mapping[names[a['category_id']]]
                    if head != 'neck':
                        raise ValueError('COCO importer supports neck polygons only; wire masks require reviewed line conversion')
                    segmentation = a.get('segmentation')
                    if not isinstance(segmentation, list) or not segmentation or a.get('iscrowd', 0):
                        raise ValueError('Expected non-crowd COCO polygon segmentation')
                    for polygon in segmentation:
                        p = np.array(polygon, float).reshape(-1,2) / [im['width'],im['height']]
                        labels.append({'head':head,'kind':'polygon','points':p.tolist()})
                image = (directory/im['file_name']).resolve()
                if not image.is_relative_to(root):
                    raise ValueError('COCO image path escapes source root')
                annotations[image] = labels
        else:
            raise ValueError('Supported formats: obb, coco')
        for image, labels in annotations.items():
            record_supervised = list(supervised)
            # In the audited B101 source a visible wire with no neck box is
            # incomplete neck supervision, not evidence that no board exists.
            masked_neck = bool(spec.get('mask_missing_neck') and 'neck' in record_supervised
                               and any(a['head'] == 'fret' for a in labels)
                               and not any(a['head'] == 'neck' for a in labels))
            if masked_neck:
                record_supervised.remove('neck')
            pixels = cv2.imread(str(image))
            if pixels is None:
                raise ValueError(f'Cannot decode {image}')
            records.append({'image':str(image.resolve()), 'split':split,
                            'group':f"{spec['name']}:{source_key(image)}", 'source':spec['name'],
                            'sha256':hashlib.sha256(image.read_bytes()).hexdigest(),
                            'phash':phash(pixels), 'supervised':record_supervised, 'annotations':labels,
                            'masked_missing_neck':masked_neck,
                            'width':pixels.shape[1], 'height':pixels.shape[0]})
    return records


def prepare(specs, output, groups=None, distance=4):
    records = [r for spec in specs for r in import_source(spec)]
    if not records:
        raise ValueError('No images found')
    if groups:
        for r in records:
            if r['group'] not in groups:
                raise ValueError(f"Missing session mapping: {r['group']}")
            r['group'] = groups[r['group']]
    counts = {}
    for split in ('train','val','test'):
        rr = [r for r in records if r['split']==split]
        counts[split] = {'images':len(rr), 'source_groups':len({r['group'] for r in rr}),
                         'annotations':dict(Counter(a['head'] for r in rr for a in r['annotations'])),
                         'empty_images':sum(not r['annotations'] for r in rr),
                         'without_nut_annotation':sum(not any(a['head']=='nut' for a in r['annotations']) for r in rr)}
    # Union both filename families and near duplicates, including across data sources.
    parent = list(range(len(records)))
    def find(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def union(i,j):
        parent[find(j)] = find(i)
    seen = {}
    near = []
    for i,r in enumerate(records):
        for key in ('group','sha256'):
            token = (key,r[key])
            if token in seen:
                union(i,seen[token])
            seen[token] = i
        for j in range(i):
            other = records[j]
            if (r['phash'] ^ other['phash']).bit_count() <= distance:
                union(i,j)
                if r['split'] != other['split']:
                    near.append([r['image'],other['image']])
    components = defaultdict(list)
    for i in range(len(records)):
        components[find(i)].append(i)
    rank = {'train':0,'val':1,'test':2}
    kept, removed = [], []
    for indices in components.values():
        # Preserve the existing test > val > train boundary. Remove lower-priority
        # duplicates, never promote augmented training examples into a held-out set.
        split = max((records[i]['split'] for i in indices), key=rank.get)
        group_id = hashlib.sha256('|'.join(sorted({records[i]['group'] for i in indices})).encode()).hexdigest()[:20]
        for i in indices:
            r = records[i]
            if r['split'] != split:
                removed.append(r['image']); continue
            r['original_group'] = r['group']; r['group'] = group_id
            kept.append(r)
    report = {'raw_splits':counts,'raw_images':len(records),'clipped_annotations':sum(a.get('clipped',False) for r in records for a in r['annotations']),'masked_missing_neck_images':sum(r['masked_missing_neck'] for r in records),'raw_source_groups':len({r['original_group'] if 'original_group' in r else r['group'] for r in records}),
              'near_duplicate_cross_split_pairs':len(near),'near_duplicate_examples':near[:30],
              'quarantined_images':len(removed),'quarantined_paths':removed,
              'prepared_splits':dict(Counter(r['split'] for r in kept)),
              'session_groups_verified':bool(groups),
              'note':'Perceptual hashes catch some overlap, not all shared sessions. Inspect video/guitar identities before interpreting held-out accuracy.'}
    doc = {'version':1,'heads':list(HEADS),'sources':specs,'audit':report,'records':kept}
    output = Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(doc,indent=2)+'\n')
    read_manifest(output)
    output.with_suffix('.audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--obb', type=Path, help='Current B101 export root')
    p.add_argument('--sources',type=Path,help='JSON list of additional source specifications')
    p.add_argument('--groups',type=Path,help='JSON source-name:filename to recording/guitar group mapping')
    p.add_argument('--output',type=Path,default=Path('data/fretboard.json'))
    p.add_argument('--hash-distance',type=int,default=4)
    args = p.parse_args()
    specs = json.loads(args.sources.read_text()) if args.sources else []
    if args.obb:
        specs.insert(0, {'name':'b101','format':'obb','root':str(args.obb.resolve()),
                        'mapping':{'neck':'neck','fret':'fret','nut':'nut'}, 'supervised':list(HEADS), 'mask_missing_neck':True,
                        'attribution':'guitar by B101, https://universe.roboflow.com/b101/guitar-g65u6',
                        'license':'CC BY 4.0 (local README.dataset.txt)'})
    if not specs or not 0 <= args.hash_distance <= 16:
        p.error('Provide --obb or --sources; hash-distance must be 0..16')
    report = prepare(specs,args.output,json.loads(args.groups.read_text()) if args.groups else None,args.hash_distance)
    print(json.dumps({k:v for k,v in report.items() if k not in ('quarantined_paths','near_duplicate_examples')},indent=2))

if __name__ == '__main__':
    main()

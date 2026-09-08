"""Train FretboardNet with partial supervision, source balancing and resumable runs."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from .data import HEADS, FretboardDataset, manifest_digest, read_manifest
from processing.vision.model import FretboardNet, masked_loss


def select_device(value):
    if value != 'auto':
        return value
    return 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed); random.seed(seed)


def atomic_save(value, path):
    temporary = path.with_suffix('.tmp')
    torch.save(value,temporary); temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('runs/dense/fretboard-v1'))
    p.add_argument('--epochs',type=int,default=150)
    p.add_argument('--imgsz',type=int,default=768)
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--accumulate',type=int,default=4)
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--patience',type=int,default=25)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='auto')
    p.add_argument('--resume',type=Path)
    p.add_argument('--from-scratch',action='store_true',help='Offline smoke tests only; default uses ImageNet encoder weights')
    p.add_argument('--max-batches',type=int,help='Smoke test only; does not establish model quality')
    args = p.parse_args()
    if min(args.epochs,args.batch,args.accumulate,args.patience) < 1 or args.imgsz < 64 or args.imgsz % 32 or args.workers < 0 or args.lr <= 0 or (args.max_batches is not None and args.max_batches < 1):
        p.error('Use positive settings; imgsz must be >=64 and divisible by 32')
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = select_device(args.device)
    # Avoid oversubscribing small local smoke tests / video inference.
    torch.set_num_threads(min(8,torch.get_num_threads()))
    digest = manifest_digest(args.data)
    doc = read_manifest(args.data)
    if not doc.get('audit',{}).get('session_groups_verified'):
        print('Dataset split is provisional: video/guitar identities have not been verified.',flush=True)
    train = FretboardDataset(args.data,'train',args.imgsz,True,args.seed)
    val = FretboardDataset(args.data,'val',args.imgsz)
    # Every source has equal expected mass; every original image has equal mass
    # within that source, regardless of the number of offline rotated copies.
    group_counts = Counter((r['source'],r['original_group']) for r in train.records)
    source_groups = Counter(source for source,group in group_counts)
    weights = [1/(source_groups[r['source']]*group_counts[(r['source'],r['original_group'])]) for r in train.records]
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(weights,len(train),replacement=True,generator=generator)
    train_loader = DataLoader(train,batch_size=args.batch,sampler=sampler,num_workers=args.workers,worker_init_fn=seed_worker,generator=generator)
    val_loader = DataLoader(val,batch_size=args.batch,num_workers=args.workers)
    model = FretboardNet(pretrained=not args.from_scratch and not args.resume).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=args.lr*.03)
    scaler = torch.GradScaler('cuda',enabled=device.startswith('cuda'))
    start, best, stale = 0, float('inf'), 0
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    if args.resume:
        state = torch.load(args.resume,map_location='cpu',weights_only=True)
        if state['manifest_sha256'] != digest:
            raise ValueError('Resume requires the same dataset manifest')
        for key in ('imgsz','batch','accumulate','epochs','seed','lr','from_scratch','max_batches'):
            if state['config'][key] != config[key]:
                raise ValueError(f'Resume requires original --{key.replace("_","-")}')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']); scaler.load_state_dict(state['scaler'])
        start,best,stale = state['epoch']+1,state['best_loss'],state['stale']
        torch.set_rng_state(state['torch_rng']); generator.set_state(state['sampler_rng'])
        if device.startswith('cuda') and state.get('cuda_rng'):
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        if args.output.resolve() != args.resume.resolve().parent:
            raise ValueError('Resume --output must be the checkpoint directory')
    elif args.output.exists():
        raise FileExistsError(f'Choose a new output directory or use --resume: {args.output}')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (args.output/'dataset-audit.json').write_text(json.dumps(doc.get('audit',{}),indent=2)+'\n')
    print(f'Training on {device}: {len(train)} train / {len(val)} val images',flush=True)
    for epoch in range(start,args.epochs):
        t0=time.perf_counter(); train.epoch=epoch; model.train(); optimizer.zero_grad(set_to_none=True)
        total, count = 0., 0
        batches=min(len(train_loader),args.max_batches or len(train_loader))
        for i,(images,target,valid) in enumerate(train_loader):
            if i>=batches: break
            images,target,valid = images.to(device),target.to(device),valid.to(device)
            accumulation=min(args.accumulate,batches-(i//args.accumulate)*args.accumulate)
            with torch.autocast(device_type='cuda',enabled=device.startswith('cuda')):
                loss=masked_loss(model(images),target,valid)
            if not torch.isfinite(loss): raise RuntimeError('Non-finite training loss')
            scaler.scale(loss/accumulation).backward()
            if (i+1)%args.accumulate==0 or i+1==batches:
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            total+=loss.item()*len(images);count+=len(images)
        model.eval(); val_total=0.;val_count=0
        with torch.inference_mode():
            for i,(images,target,valid) in enumerate(val_loader):
                if args.max_batches and i>=args.max_batches: break
                loss=masked_loss(model(images.to(device)),target.to(device),valid.to(device))
                if not torch.isfinite(loss): raise RuntimeError('Non-finite validation loss')
                val_total+=loss.item()*len(images);val_count+=len(images)
        val_loss=val_total/val_count
        improved=val_loss<best
        best=min(best,val_loss);stale=0 if improved else stale+1
        scheduler.step()
        state={'architecture':'fretboard-resnet34-unet-v1','heads':list(HEADS),'config':config,
               'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
               'scaler':scaler.state_dict(),'epoch':epoch,'best_loss':best,'stale':stale,
               'manifest_sha256':digest,'torch_rng':torch.get_rng_state(),'sampler_rng':generator.get_state(),
               'cuda_rng':torch.cuda.get_rng_state_all() if device.startswith('cuda') else [],
               'smoke_test':bool(args.max_batches),'torch_version':str(torch.__version__)}
        atomic_save(state,args.output/'last.pt')
        if improved: atomic_save(state,args.output/'best.pt')
        row={'epoch':epoch+1,'train_loss':total/count,'val_loss':val_loss,'seconds':time.perf_counter()-t0,'lr':scheduler.get_last_lr()[0]}
        with (args.output/'metrics.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        if stale>=args.patience: break
    print(f"Checkpoint: {args.output/'best.pt'}",flush=True)

if __name__=='__main__': main()

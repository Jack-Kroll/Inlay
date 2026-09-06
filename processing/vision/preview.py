"""Dense fretboard diagnostics: observations, heatmaps, tracking state and JSONL."""
import argparse
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from .tracker import FretboardTracker, decode_maps

def open_capture(source):
    """Use the macOS camera backend explicitly and report actionable failures."""
    is_camera = isinstance(source, int)
    cap = (cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
           if is_camera and sys.platform == "darwin" else cv2.VideoCapture(source))
    if cap.isOpened():
        return cap
    cap.release()
    if is_camera and sys.platform == "darwin":
        raise RuntimeError(
            f"Cannot open camera {source}. If macOS requested access, allow it and rerun.\n"
            "Otherwise open System Settings > Privacy & Security > Camera and enable\n"
            "the app running this command (Terminal, VS Code, or Codex). Restart that\n"
            "app and rerun. If access is already enabled, close other camera apps\n"
            "and check the camera index (--source 0)."
        )
    raise RuntimeError(f"Cannot open {'camera' if is_camera else 'video'} {source!r}. "
                       "Check the device/permissions or file path.")


def draw_text(frame, text, xy, scale=0.55, color=(255, 255, 255), thickness=1):
    x, y = int(xy[0]), int(xy[1])
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)



def draw_observation(frame,observation,state):
    if observation is None:return
    color=(0,210,0) if state=='detected' else (0,180,255)
    cv2.polylines(frame,[observation.neck.astype(np.int32)],True,color,2)
    for line,number in zip(observation.frets,observation.numbers):
        cv2.line(frame,tuple(line[0].astype(int)),tuple(line[1].astype(int)),(255,220,0),2)
        if number is not None:draw_text(frame,str(number),line.mean(axis=0),.4)
    if observation.nut is not None:
        cv2.line(frame,tuple(observation.nut[0].astype(int)),tuple(observation.nut[1].astype(int)),(255,0,255),3)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True,type=Path)
    ap.add_argument('--source',default='0')
    ap.add_argument('--device',default='auto')
    ap.add_argument('--imgsz',type=int)
    ap.add_argument('--threshold',type=float,default=.5)
    ap.add_argument('--max-gap',type=float,default=.5)
    ap.add_argument('--show-heatmaps','--show-detections',dest='show_heatmaps',action='store_true')
    ap.add_argument('--jsonl',type=Path)
    ap.add_argument('--headless',action='store_true')
    ap.add_argument('--max-frames',type=int)
    args=ap.parse_args()
    if not 0<args.threshold<1 or args.max_gap<=0 or (args.max_frames is not None and args.max_frames<1):ap.error('Invalid threshold, max-gap or max-frames')
    from .model import DenseDetector
    from processing.training.train import select_device
    detector=DenseDetector(args.model,select_device(args.device),args.imgsz)
    tracker=FretboardTracker(max_gap=args.max_gap)
    source=int(args.source) if str(args.source).isdigit() else args.source
    cap=open_capture(source);output=None
    try:
        if args.jsonl:
            args.jsonl.parent.mkdir(parents=True,exist_ok=True);output=args.jsonl.open('x')
        fps=cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps<=0:fps=30.
        origin=time.monotonic();index=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            timestamp=time.monotonic()-origin if isinstance(source,int) else index/fps
            t0=time.perf_counter();maps=detector.predict(frame)
            observation=tracker.update(frame,decode_maps(maps,args.threshold),timestamp)
            latency=(time.perf_counter()-t0)*1000
            record={'frame':index,'timestamp':timestamp,'state':tracker.state,'latency_ms':latency,
                    'flow_inliers':tracker.flow_inliers,'numbering':observation.numbering if observation else 'unknown',
                    'neck':observation.neck.tolist() if observation else None,
                    'frets':[{'endpoints':line.tolist(),'number':number} for line,number in zip(observation.frets,observation.numbers)] if observation else [],
                    'confidence':observation.confidence if observation else 0.}
            if output:output.write(json.dumps(record)+'\n')
            if not args.headless:
                draw_observation(frame,observation,tracker.state)
                draw_text(frame,f'{tracker.state} | {latency:.0f} ms | numbering: {record["numbering"]}',(15,25))
                if args.show_heatmaps:
                    for i,name in enumerate(('neck','fret','nut')):
                        heat=cv2.applyColorMap((maps[...,i]*255).astype(np.uint8),cv2.COLORMAP_INFERNO)
                        cv2.imshow(name,heat)
                cv2.imshow('fretboard',frame)
                key=cv2.waitKey(1)&0xff
                if key==ord('q'):break
                if key==ord('r'):tracker.reset()
            index+=1
            if args.max_frames and index>=args.max_frames:break
    finally:
        cap.release()
        if output:output.close()
        if not args.headless:cv2.destroyAllWindows()

if __name__=='__main__':main()

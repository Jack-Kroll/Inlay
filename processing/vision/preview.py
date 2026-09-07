"""Dense fretboard diagnostics: observations, heatmaps, tracking state and JSONL."""
from collections import deque
import threading
import argparse
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from .tracker import FretboardTracker, decode_maps, transform_observation

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




class LatestCameraFrame:
    """Drain the live camera continuously; inference consumes the newest frame.

    Video files remain synchronous so recorded-frame evaluation skips nothing.
    The worker owns capture.release(), avoiding concurrent read/release calls.
    """
    def __init__(self, capture, origin):
        self.capture = capture
        self.origin = origin
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.latest = None
        self.finished = False
        self.error = None
        self.worker = threading.Thread(target=self._run, name='inlay-camera', daemon=True)
        self.worker.start()

    def _run(self):
        index = 0
        try:
            while not self.stop.is_set():
                ok, frame = self.capture.read()
                if not ok:
                    break
                timestamp = time.monotonic() - self.origin
                with self.condition:
                    self.latest = (index, timestamp, frame)
                    self.condition.notify_all()
                index += 1
        except Exception as error:
            self.error = error
        finally:
            self.capture.release()
            with self.condition:
                self.finished = True
                self.condition.notify_all()

    def read(self, timeout=3.0):
        with self.condition:
            self.condition.wait_for(
                lambda: self.latest is not None or self.finished or self.stop.is_set(),
                timeout=timeout,
            )
            packet, self.latest = self.latest, None
            if packet is None and self.error is not None:
                raise RuntimeError('Camera capture failed') from self.error
            return packet

    def close(self):
        self.stop.set()
        with self.condition:
            self.condition.notify_all()
        self.worker.join(timeout=2.0)


def display_observation(observation, width, mirrored):
    """Return display coordinates without changing tracker/JSONL coordinates."""
    if observation is None or not mirrored:
        return observation
    matrix = np.array([[-1., 0., width-1.], [0., 1., 0.], [0., 0., 1.]])
    return transform_observation(observation, matrix)


def draw_observation(frame, observation, state):
    if observation is None:
        return
    color = (0, 210, 0) if state == 'detected' else (0, 180, 255)
    cv2.polylines(frame, [observation.neck.astype(np.int32)], True, color, 2)
    estimated_color = (0, 165, 255)
    for estimate in observation.estimates:
        line = estimate.line
        cv2.line(frame, tuple(line[0].astype(int)), tuple(line[1].astype(int)), estimated_color, 2)
        draw_text(frame, f'~{estimate.number}', line.mean(axis=0), .4, color=estimated_color)
    line_color = (255, 220, 0) if state == 'detected' else estimated_color
    for line, number in zip(observation.frets, observation.numbers):
        cv2.line(frame, tuple(line[0].astype(int)), tuple(line[1].astype(int)), line_color, 2)
        if number is not None:
            label = str(number) if state == 'detected' else f'~{number}'
            draw_text(frame, label, line.mean(axis=0), .4)
    if observation.nut is not None:
        cv2.line(frame, tuple(observation.nut[0].astype(int)),
                 tuple(observation.nut[1].astype(int)), (255, 0, 255), 3)


def render_preview(frame, observation, state, mirrored=True):
    display = cv2.flip(frame, 1) if mirrored else frame.copy()
    draw_observation(display, display_observation(observation, frame.shape[1], mirrored), state)
    return display


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True, type=Path)
    ap.add_argument('--source', default='0')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--imgsz', type=int, help='Inference size; default keeps the trained size. Try 640 for extra speed.')
    ap.add_argument('--threshold', type=float, default=.5)
    ap.add_argument('--max-gap', type=float, default=.5)
    ap.add_argument('--mirror', action=argparse.BooleanOptionalAction, default=True,
                    help='Mirror the display like a front camera (default). Raw coordinates stay unchanged.')
    ap.add_argument('--show-heatmaps', '--show-detections', dest='show_heatmaps', action='store_true')
    ap.add_argument('--heatmap-fps', type=float, default=5.,
                    help='Refresh diagnostic maps separately from the camera preview')
    ap.add_argument('--jsonl', type=Path)
    ap.add_argument('--headless', action='store_true')
    ap.add_argument('--max-frames', type=int)
    args = ap.parse_args()
    if (not 0 < args.threshold < 1 or not np.isfinite(args.max_gap) or args.max_gap <= 0
            or not np.isfinite(args.heatmap_fps) or args.heatmap_fps <= 0
            or (args.max_frames is not None and args.max_frames < 1)):
        ap.error('Invalid threshold, max-gap, heatmap-fps or max-frames')
    from .model import DenseDetector
    from processing.training.train import select_device
    detector = DenseDetector(args.model, select_device(args.device), args.imgsz)
    tracker = FretboardTracker(max_gap=args.max_gap)
    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = open_capture(source)
    output = None
    camera = None
    try:
        if args.jsonl:
            args.jsonl.parent.mkdir(parents=True, exist_ok=True)
            output = args.jsonl.open('x')
        if isinstance(source, int):
            # Backends may decline these hints; latest-frame capture still avoids a queue.
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.
        origin = time.monotonic()
        if isinstance(source, int):
            camera = LatestCameraFrame(cap, origin)
        index = 0
        recent_frames = deque(maxlen=30)
        last_heatmap = -float('inf')
        while True:
            if camera is not None:
                packet = camera.read()
                if packet is None:
                    break
                capture_index, timestamp, frame = packet
            else:
                ok, frame = cap.read()
                if not ok:
                    break
                capture_index, timestamp = index, index/fps
            t0 = time.perf_counter()
            recent_frames.append(t0)
            actual_fps = ((len(recent_frames)-1)/(recent_frames[-1]-recent_frames[0])
                          if len(recent_frames)>1 and recent_frames[-1]>recent_frames[0] else None)
            maps = detector.predict(frame)
            t1 = time.perf_counter()
            detected = decode_maps(maps, args.threshold)
            t2 = time.perf_counter()
            observation = tracker.update(frame, detected, timestamp)
            t3 = time.perf_counter()
            latency = (t3-t0)*1000
            record = {
                'frame': index, 'capture_frame': capture_index, 'timestamp': timestamp,
                'state': tracker.state, 'latency_ms': latency, 'fps': actual_fps,
                'inference_ms': (t1-t0)*1000, 'decode_ms': (t2-t1)*1000,
                'tracking_ms': (t3-t2)*1000,
                'flow_inliers': tracker.flow_inliers,
                'numbering': observation.numbering if observation else 'unknown',
                'neck': observation.neck.tolist() if observation else None,
                'frets': [{'endpoints': line.tolist(), 'number': number,
                           'source': 'detected' if tracker.state == 'detected' else 'tracked'}
                          for line, number in zip(observation.frets, observation.numbers)] if observation else [],
                'estimated_frets': [{'endpoints': e.line.tolist(), 'number': e.number,
                                     'source': 'estimated', 'method': e.method}
                                    for e in observation.estimates] if observation else [],
                'confidence': observation.confidence if observation else 0.,
            }
            if output:
                output.write(json.dumps(record)+'\n')
            if not args.headless:
                display = render_preview(frame, observation, tracker.state, args.mirror)
                fps_text = f'{actual_fps:.1f} FPS' if actual_fps is not None else 'warming up'
                draw_text(display, f'{tracker.state} | {fps_text} | {latency:.0f} ms | numbering: {record["numbering"]}', (15, 25))
                draw_text(display, 'Blue: detected | Orange: estimated (~)', (15, 47), .45)
                if args.show_heatmaps and t3-last_heatmap >= 1/args.heatmap_fps:
                    scale = min(1., 480/maps.shape[1])
                    small = cv2.resize(maps, (max(1, round(maps.shape[1]*scale)),
                                              max(1, round(maps.shape[0]*scale))))
                    if args.mirror:
                        small = cv2.flip(small, 1)
                    for i, name in enumerate(('neck', 'fret', 'nut')):
                        heat = cv2.applyColorMap((small[..., i]*255).astype(np.uint8), cv2.COLORMAP_INFERNO)
                        cv2.imshow(name, heat)
                    last_heatmap = t3
                cv2.imshow('fretboard', display)
                key = cv2.waitKey(1) & 0xff
                if key == ord('q'):
                    break
                if key == ord('r'):
                    tracker.reset()
                if key == ord('m'):
                    args.mirror = not args.mirror
                    last_heatmap = -float('inf')
            index += 1
            if args.max_frames and index >= args.max_frames:
                break
    finally:
        if camera is not None:
            camera.close()
        else:
            cap.release()
        if output:
            output.close()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()



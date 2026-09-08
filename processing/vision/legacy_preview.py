"""OBB overlay: uv run --extra legacy python -m processing.vision.legacy_preview."""
import argparse
import sys
from collections import deque
import cv2
import numpy as np
from .geometry import (
    CLASS_NAMES, INLAY_FRETS, SCALE_NUT_RATIO, MAX_A_JUMP_FRAC,
    obb_to_segment, extract_detections, normalize, proj_along,
    select_anchors, smooth_vec, invert_projective, fit_from_assigned,
    assign_from_prior, fit_from_search, fit_rail, rail_at, fret_ratio,
)

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


def main():
    ap = argparse.ArgumentParser(description="Perspective-correct fretboard grid overlay")
    ap.add_argument("--model",           required=True,            help="Path to best.pt")
    ap.add_argument("--source",          default="0",              help="Camera index or video path")
    ap.add_argument("--conf",            type=float, default=0.15)
    ap.add_argument("--imgsz",           type=int,   default=640)
    ap.add_argument("--anchor-conf",     type=float, default=0.30, help="Min conf to use a fret as anchor")
    ap.add_argument("--max-anchors",     type=int,   default=5)
    ap.add_argument("--min-sep",         type=float, default=18.0, help="Min px between anchors")
    ap.add_argument("--max-fret",        type=int,   default=22)
    ap.add_argument("--smooth",          type=int,   default=8,    help="Temporal smoothing window (frames)")
    ap.add_argument("--show-detections", action="store_true",      help="Draw raw OBB detections")
    args = ap.parse_args()
    if args.smooth < 1 or args.max_anchors < 2 or args.max_fret < 1:
        ap.error("smooth and max-fret must be positive; max-anchors must be at least 2")

    from ultralytics import YOLO

    model  = YOLO(args.model)
    source = int(args.source) if str(args.source).isdigit() else args.source
    try:
        cap = open_capture(source)
    except RuntimeError as error:
        ap.exit(1, f"{error}\n")

    N = args.smooth
    nut_center_h  = deque(maxlen=N)
    axis_dir_h    = deque(maxlen=N)
    fret_u_h      = deque(maxlen=N)
    fit_h         = deque(maxlen=N)
    left_base_h   = deque(maxlen=N)
    left_slope_h  = deque(maxlen=N)
    right_base_h  = deque(maxlen=N)
    right_slope_h = deque(maxlen=N)

    print("fretgrid — press q to quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
            names  = getattr(model, "names", CLASS_NAMES)
            dets   = extract_detections(result, names)

            if not dets["nut"]:
                for history in (nut_center_h, axis_dir_h, fret_u_h, fit_h, left_base_h, left_slope_h, right_base_h, right_slope_h):
                    history.clear()
                draw_text(frame, "no nut detected", (20, 35), color=(0, 0, 255))
                cv2.imshow("fretgrid", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            nut_det = max(dets["nut"], key=lambda d: d["conf"])
            nut_seg = obb_to_segment(nut_det["poly"])

            # Parse fret OBBs. obb_to_segment already ensures p1=left/p2=right.
            fret_segs = []
            for d in dets["fret"]:
                fs = obb_to_segment(d["poly"])
                fs["conf"]     = d["conf"]
                fs["raw_poly"] = d["poly"]
                fret_segs.append(fs)

            # --- Geometry estimation ---
            good_dirs = [nut_seg["u"]] + [fs["u"] for fs in fret_segs if fs["conf"] >= args.anchor_conf]
            fret_u    = normalize(np.mean(good_dirs, axis=0))
            axis_dir  = normalize(np.array([-fret_u[1], fret_u[0]]))

            projs = [proj_along(fs["center"], nut_seg["center"], axis_dir) for fs in fret_segs]
            if projs and float(np.median(projs)) < 0:
                axis_dir = -axis_dir

            anchors = select_anchors(fret_segs, nut_seg["center"], axis_dir,
                                     args.anchor_conf, args.max_anchors, args.min_sep)

            # Refine axis direction from nut + anchor centerline fit
            if len(anchors) >= 2:
                cpts = np.array(
                    [nut_seg["center"]] + [a["seg"]["center"] for a in anchors],
                    dtype=np.float32,
                ).reshape(-1, 1, 2)
                ln = cv2.fitLine(cpts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
                d  = normalize(np.array([float(ln[0]), float(ln[1])]))
                if np.dot(d, axis_dir) < 0:
                    d = -d
                axis_dir = d

            # Smooth geometry
            nut_center_h.append(nut_seg["center"])
            axis_dir_h.append(axis_dir)
            fret_u_h.append(fret_u)
            nut_center = smooth_vec(nut_center_h, nut_seg["center"])
            axis_dir   = normalize(smooth_vec(axis_dir_h, axis_dir))
            fret_u     = normalize(smooth_vec(fret_u_h,   fret_u))

            anchors = select_anchors(fret_segs, nut_center, axis_dir,
                                     args.anchor_conf, args.max_anchors, args.min_sep)

            # --- Projective fit ---
            nut_width_px = nut_seg["length"]
            a_expected   = nut_width_px * SCALE_NUT_RATIO
            max_frets    = max(24, args.max_fret)

            fit  = None
            mode = "frozen"
            if len(anchors) >= 2:
                if fit_h:
                    # Primary path: invert the smoothed fit to assign fret numbers, then refit.
                    # Works for any visible frets, not just the first 5.
                    prev_a = float(np.mean([f["a"] for f in fit_h]))
                    prev_c = float(np.mean([f["c"] for f in fit_h]))
                    assigned = []
                    for anc in anchors:
                        n_float = invert_projective(anc["s"], prev_a, prev_c)
                        if n_float is not None:
                            n = int(round(n_float))
                            if 1 <= n <= max_frets and abs(n_float - n) < 0.45:
                                assigned.append({**anc, "fret_num": n})
                    if len(assigned) >= 2:
                        fit  = fit_from_assigned(assigned, max_frets)
                        mode = "forward"

                if fit is None:
                    # Cold-start: invert equal-temperament using nut-width prior (c≈0 assumption).
                    # Assigns fret numbers to ANY detected frets — no consecutive-fret assumption.
                    assigned = assign_from_prior(anchors, a_expected, max_frets)
                    if len(assigned) >= 2:
                        fit  = fit_from_assigned(assigned, max_frets)
                        mode = "prior"

                if fit is None:
                    # Last resort: search over consecutive start-fret candidates.
                    fit  = fit_from_search(anchors, max_frets, a_expected)
                    mode = "search"

                # Temporal stability clamp: reject if scale length jumps wildly.
                if fit is not None and fit_h:
                    prev_a = float(np.mean([f["a"] for f in fit_h]))
                    jump   = abs(fit["a"] - prev_a) / max(prev_a, 1.0)
                    if jump > MAX_A_JUMP_FRAC:
                        fit  = None
                        mode = "frozen"

            if fit:
                fit_h.append(fit)

            smooth_fit = None
            if fit_h:
                smooth_fit = {
                    "a":     float(np.mean([f["a"]    for f in fit_h])),
                    "c":     float(np.mean([f["c"]    for f in fit_h])),
                    "rmse":  float(np.mean([f["rmse"] for f in fit_h])),
                    "start": int(round(np.mean([f["start"] for f in fit_h]))),
                }

            # --- Rail fitting ---
            # p1 = left (smaller image-x), p2 = right — guaranteed by obb_to_segment
            left_pts  = [nut_seg["p1"]] + [a["seg"]["p1"] for a in anchors]
            right_pts = [nut_seg["p2"]] + [a["seg"]["p2"] for a in anchors]

            left_rail  = fit_rail(left_pts,  nut_center, axis_dir)
            right_rail = fit_rail(right_pts, nut_center, axis_dir)

            if left_rail:
                left_base_h.append(left_rail[0])
                left_slope_h.append(left_rail[1])
            if right_rail:
                right_base_h.append(right_rail[0])
                right_slope_h.append(right_rail[1])

            left_base   = smooth_vec(left_base_h)
            left_slope  = smooth_vec(left_slope_h)
            right_base  = smooth_vec(right_base_h)
            right_slope = smooth_vec(right_slope_h)
            have_rails  = left_base is not None and right_base is not None

            # ---- Draw raw detections ----
            if args.show_detections:
                cv2.polylines(frame, [nut_det["poly"].astype(int)], True, (255, 0, 255), 2)
                anchor_ids = {id(a["seg"]) for a in anchors}
                for fs in fret_segs:
                    col   = (0, 255, 0) if id(fs) in anchor_ids else (0, 80, 0)
                    thick = 2 if id(fs) in anchor_ids else 1
                    cv2.polylines(frame, [fs["raw_poly"].astype(int)], True, col, thick)
                    cv2.circle(frame, tuple(fs["p1"].astype(int)), 5, (255, 140,   0), -1)  # orange = left
                    cv2.circle(frame, tuple(fs["p2"].astype(int)), 5, (  0, 140, 255), -1)  # blue   = right

            # ---- Draw grid ----
            if have_rails:
                pL_nut = rail_at(left_base,  left_slope,  0.0)
                pR_nut = rail_at(right_base, right_slope, 0.0)
            else:
                pL_nut = nut_seg["p1"]
                pR_nut = nut_seg["p2"]

            cv2.line(frame, tuple(pL_nut.astype(int)), tuple(pR_nut.astype(int)), (255, 0, 255), 3)
            draw_text(frame, "nut", nut_center + np.array([6, -6]), color=(255, 0, 255), scale=0.45)

            prev_pL, prev_pR = pL_nut, pR_nut
            if smooth_fit and have_rails:
                a, c = smooth_fit["a"], smooth_fit["c"]

                for n in range(1, args.max_fret + 1):
                    r     = fret_ratio(n)
                    denom = c * r + 1.0
                    if abs(denom) < 1e-8:
                        continue
                    s  = (a * r) / denom
                    pL = rail_at(left_base,  left_slope,  s)
                    pR = rail_at(right_base, right_slope, s)

                    thick = 2 if n == 12 else 1
                    cv2.line(frame, tuple(pL.astype(int)), tuple(pR.astype(int)), (255, 255, 0), thick)

                    if n in INLAY_FRETS:
                        mid = (pL + pR) / 2.0
                        draw_text(frame, str(n), mid + np.array([4, -3]), color=(255, 255, 0), scale=0.42)

                    prev_pL, prev_pR = pL, pR

                # 6 strings
                for i in range(6):
                    t  = (i + 0.5) / 6.0
                    p0 = pL_nut  * (1 - t) + pR_nut  * t
                    p1 = prev_pL * (1 - t) + prev_pR * t
                    cv2.line(frame, tuple(p0.astype(int)), tuple(p1.astype(int)), (170, 170, 170), 1)

                # Fretboard outline
                outline = np.array([pL_nut, pR_nut, prev_pR, prev_pL], dtype=np.int32)
                cv2.polylines(frame, [outline], True, (0, 200, 255), 2)

            status = f"anchors {len(anchors)} | {mode}"
            if smooth_fit:
                status += f" | rmse {smooth_fit['rmse']:.1f} | start~{smooth_fit['start']}"
            else:
                status += " | awaiting fit"
            draw_text(frame, status, (20, 30))

            cv2.imshow("fretgrid", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

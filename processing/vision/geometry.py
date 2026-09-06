"""Image-space fretboard geometry. No camera, UI, or ML runtime dependency."""
import math
import cv2
import numpy as np

CLASS_NAMES = {0: "fret", 1: "neck", 2: "nut"}
INLAY_FRETS = {1, 3, 5, 7, 9, 12, 15, 17, 19, 21}

# Typical guitar: scale length ≈ 15× nut width. Used as a rough prior to filter bad fits.
SCALE_NUT_RATIO  = 15.0
A_PRIOR_BAND     = 4.0   # accept a_fit in [a_expected/4, a_expected*4]
MAX_A_JUMP_FRAC  = 0.50  # reject fit if `a` changes by more than this fraction from history


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def fret_ratio(n: int) -> float:
    return 1.0 - 1.0 / (2.0 ** (n / 12.0))


def normalize(v, eps: float = 1e-8):
    mag = np.linalg.norm(v)
    return v / mag if mag >= eps else np.zeros_like(v)


def proj_along(pt, origin, axis) -> float:
    return float(np.dot(np.asarray(pt, np.float64) - np.asarray(origin, np.float64), axis))


# ---------------------------------------------------------------------------
# OBB → segment
# ---------------------------------------------------------------------------

def obb_to_segment(poly):
    """
    Thin OBB → centerline segment.

    p1 is ALWAYS the endpoint with the smaller image-x coordinate.
    This is the key stability guarantee: left/right assignment never flips
    between frames regardless of which way the model's edge directions point.
    """
    p = np.asarray(poly, dtype=np.float64).reshape(4, 2)
    edges = sorted(
        [(i, (i + 1) % 4, np.linalg.norm(p[(i + 1) % 4] - p[i])) for i in range(4)],
        key=lambda e: -e[2],
    )
    d1 = normalize(p[edges[0][1]] - p[edges[0][0]])
    d2 = normalize(p[edges[1][1]] - p[edges[1][0]])
    if np.dot(d1, d2) < 0:
        d2 = -d2
    u      = normalize(d1 + d2)
    center = p.mean(axis=0)
    length = (edges[0][2] + edges[1][2]) / 2.0
    p1 = center - 0.5 * length * u
    p2 = center + 0.5 * length * u
    # Canonical left = smaller x. If neck is near-vertical, fall back to smaller y.
    if abs(u[0]) >= abs(u[1]):
        swap = p1[0] > p2[0]
    else:
        swap = p1[1] > p2[1]
    if swap:
        p1, p2, u = p2, p1, -u
    return {"center": center, "p1": p1, "p2": p2, "u": u, "length": length}


# ---------------------------------------------------------------------------
# Detection parsing
# ---------------------------------------------------------------------------

def extract_detections(result, names):
    out = {"fret": [], "nut": [], "neck": []}
    if result.obb is None or result.obb.xyxyxyxy is None:
        return out
    polys = result.obb.xyxyxyxy.cpu().numpy()
    clss  = result.obb.cls.cpu().numpy().astype(int)
    confs = (result.obb.conf.cpu().numpy()
             if result.obb.conf is not None else np.ones(len(clss)))
    for poly, cls_id, conf in zip(polys, clss, confs):
        name = names.get(cls_id, str(cls_id))
        if name in out:
            out[name].append({
                "poly": poly.reshape(4, 2).astype(np.float32),
                "conf": float(conf),
            })
    return out


# ---------------------------------------------------------------------------
# Anchor selection
# ---------------------------------------------------------------------------

def select_anchors(fret_segs, nut_center, axis_dir, min_conf, max_n, min_sep):
    candidates = []
    for fs in fret_segs:
        if fs["conf"] < min_conf:
            continue
        s = proj_along(fs["center"], nut_center, axis_dir)
        if s <= 1.0:
            continue
        candidates.append((s, fs))
    candidates.sort(key=lambda x: x[0])
    deduped = []
    for s, fs in candidates:
        if not deduped or abs(s - deduped[-1][0]) >= min_sep:
            deduped.append((s, fs))
        elif fs["conf"] > deduped[-1][1]["conf"]:
            deduped[-1] = (s, fs)
    return [{"s": s, "conf": fs["conf"], "seg": fs} for s, fs in deduped[:max_n]]


# ---------------------------------------------------------------------------
# Projective fit  s = a*r / (c*r + 1)
# ---------------------------------------------------------------------------

def _lstsq_fit(ss, ws, rs, max_frets):
    """Core least-squares solve. Returns (a, c) or None."""
    W = np.sqrt(ws)[:, None]
    A = np.column_stack([rs, -ss * rs])
    try:
        sol, _, rank, _ = np.linalg.lstsq(A * W, ss * W[:, 0], rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < 2 or not np.all(np.isfinite(sol)):
        return None
    a, c = float(sol[0]), float(sol[1])
    if a <= 0:
        return None
    denom = c * rs + 1.0
    if np.any(np.abs(denom) < 1e-8):
        return None
    # Monotonicity across full fret range
    all_rs = np.array([fret_ratio(n) for n in range(1, max_frets + 1)])
    all_denoms = c * all_rs + 1.0
    if np.any(all_denoms <= 1e-8):
        return None
    all_s  = (a * all_rs) / all_denoms
    if np.any(np.diff(all_s) <= 0):
        return None
    pred = (a * rs) / denom
    rmse = float(np.sqrt(np.average((pred - ss) ** 2, weights=ws)))
    return a, c, rmse


def invert_projective(s, a, c):
    """
    Given pixel distance s from nut, return fractional fret number.
    Inverts s = a*r/(c*r+1)  →  r = s/(a - c*s)  →  n = -12*log2(1-r)
    Returns None if outside valid range.
    """
    denom = a - c * s
    if abs(denom) < 1e-8 or denom <= 0:
        return None
    r = s / denom
    if r <= 0 or r >= 1.0:
        return None
    try:
        return -12.0 * math.log2(1.0 - r)
    except ValueError:
        return None


def fit_from_assigned(anchors, max_frets):
    """Fit when each anchor has a known 'fret_num'. Anchors don't need to be consecutive."""
    if len(anchors) < 2:
        return None
    if len({a["fret_num"] for a in anchors}) < 2:
        return None
    ss = np.array([a["s"]              for a in anchors], dtype=np.float64)
    ws = np.array([max(0.05, a["conf"]) for a in anchors], dtype=np.float64)
    rs = np.array([fret_ratio(a["fret_num"]) for a in anchors], dtype=np.float64)
    result = _lstsq_fit(ss, ws, rs, max_frets)
    if result is None:
        return None
    a, c, rmse = result
    return {"a": a, "c": c, "rmse": rmse,
            "start": min(a["fret_num"] for a in anchors)}


def assign_from_prior(anchors, a_expected, max_frets, tol=0.45):
    """
    Cold-start fret assignment using the nut-width scale prior.

    Assumes c≈0 so that s ≈ a*r, giving r = s/a_expected directly.
    Inverts equal-temperament: n = -12*log2(1-r).

    Works for ANY detected frets regardless of which ones or whether they are
    consecutive — no start-candidate search needed.
    """
    assigned = []
    for anc in anchors:
        s = anc["s"]
        if s <= 0 or s >= a_expected:
            continue
        r = s / a_expected
        if r <= 0 or r >= 1.0:
            continue
        try:
            n_float = -12.0 * math.log2(1.0 - r)
        except ValueError:
            continue
        n = int(round(n_float))
        if 1 <= n <= max_frets and abs(n_float - n) < tol:
            assigned.append({**anc, "fret_num": n})
    return assigned


def fit_from_search(anchors, max_frets, a_expected):
    """
    Last-resort fallback: search over consecutive start-fret candidates.
    Only reached if both history-inversion and prior-assignment fail.
    """
    if len(anchors) < 2:
        return None
    ss = np.array([a["s"]              for a in anchors], dtype=np.float64)
    ws = np.array([max(0.05, a["conf"]) for a in anchors], dtype=np.float64)
    best = None
    for start in range(1, 20):
        nums = list(range(start, start + len(anchors)))
        if nums[-1] > max_frets:
            break
        rs = np.array([fret_ratio(n) for n in nums], dtype=np.float64)
        result = _lstsq_fit(ss, ws, rs, max_frets)
        if result is None:
            continue
        a, c, rmse = result
        # Filter by nut-width prior
        if a < a_expected / A_PRIOR_BAND or a > a_expected * A_PRIOR_BAND:
            continue
        if best is None or rmse < best["rmse"]:
            best = {"a": a, "c": c, "rmse": rmse, "start": start}
    return best


# ---------------------------------------------------------------------------
# Rail fitting
# ---------------------------------------------------------------------------

def fit_rail(pts_2d, nut_center, axis_dir):
    """
    Fit a straight line through 2D points.
    Returns (base_pt, slope_vec) where rail(s) = base_pt + s * slope_vec.
    base_pt  = point on the rail at axial distance 0 (nut level).
    slope_vec = displacement per unit axial distance.
    This form is smooth-friendly and avoids sign-flip issues.
    """
    if len(pts_2d) < 2:
        return None
    pts  = np.array(pts_2d, dtype=np.float32).reshape(-1, 1, 2)
    line = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    vx, vy, x0, y0 = float(line[0]), float(line[1]), float(line[2]), float(line[3])
    axial_dot = vx * axis_dir[0] + vy * axis_dir[1]
    if abs(axial_dot) < 1e-8:
        return None
    if axial_dot < 0:
        vx, vy, axial_dot = -vx, -vy, -axial_dot
    slope_vec  = np.array([vx / axial_dot, vy / axial_dot], dtype=np.float64)
    curr_axial = (x0 - nut_center[0]) * axis_dir[0] + (y0 - nut_center[1]) * axis_dir[1]
    base_pt    = np.array([x0, y0], dtype=np.float64) - curr_axial * slope_vec
    return base_pt, slope_vec


def rail_at(base, slope, s):
    return base + s * slope


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def smooth_vec(history, default=None):
    if not history:
        return default
    return np.mean(np.array(list(history), dtype=np.float64), axis=0)


# ---------------------------------------------------------------------------

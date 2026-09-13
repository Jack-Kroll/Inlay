"""Turn heatmaps into fretboard geometry, number the frets, and track it over time.

Pipeline per frame:
  1. ``decode_maps`` reads the neck / fret / nut heatmaps and fits straight lines.
  2. ``assign_numbers`` works out which fret each line is (1, 2, 3...) from the
     way fret spacing shrinks along the neck, anchored at the nut.
  3. ``FretboardTracker`` smooths that geometry between frames and bridges short
     gaps (a hand covering the board) using optical flow.

Nothing here relies on decorative markers such as inlay dots.
"""
from dataclasses import dataclass, field
from itertools import combinations
import cv2
import numpy as np


@dataclass
class EstimatedFret:
    """A fret we did not see but inferred, either from spacing or from tracking."""
    line: np.ndarray  # 2x2 array: the wire's two endpoints in pixels.
    number: int
    method: str = 'spacing'  # 'spacing' (interpolated) or 'tracked' (carried by motion).


@dataclass
class Observation:
    """Everything we know about the fretboard in one frame, in image pixels."""
    neck: np.ndarray                 # Convex outline of the neck, Nx2.
    frets: list[np.ndarray]          # Detected fret wires, each a 2x2 segment.
    nut: np.ndarray | None           # The nut segment, if seen.
    confidence: float                # Mean neck heatmap score inside the outline.
    numbers: list[int | None] = field(default_factory=list)  # Fret number per wire.
    numbering: str = 'unknown'       # How numbers were obtained (fit / tracked / unknown).
    estimates: list[EstimatedFret] = field(default_factory=list)  # Inferred, never measured.


def decode_maps(maps, threshold=.5):
    """Convert the model's HxWx3 heatmaps into an ``Observation``.

    Returns None if there is no convincing neck in the frame.
    """
    if maps.ndim != 3 or maps.shape[2] != 3 or not np.isfinite(maps).all():
        raise ValueError('Expected finite HxWx3 probability maps')

    # --- Neck: take the largest blob in the neck heatmap. ---
    binary = (maps[..., 0] > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < max(64, maps.shape[0] * maps.shape[1] * .001):
        return None  # Too small to be a real neck.
    neck = cv2.convexHull(contour).reshape(-1, 2).astype(np.float32)

    # --- Board axis: the long side of the tightest rotated box around the neck. ---
    rect = cv2.minAreaRect(neck)
    box = cv2.boxPoints(rect)
    edges = np.roll(box, -1, axis=0) - box
    axis = edges[np.argmax(np.linalg.norm(edges, axis=1))]
    axis /= np.linalg.norm(axis)
    # Fix the sign so the axis does not flip direction between frames.
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    width = min(rect[1])
    length = max(rect[1])
    if width < 4 or length < 2 * width:
        return None  # A neck is long and thin; reject squat blobs.

    # Region to search for wires: the neck, grown a little so wires that stick
    # slightly past the edge still count. The mask only says *where* the neck
    # is; it must not decide where wires end.
    mask = np.zeros(maps.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, neck.astype(np.int32), 1)
    padding = max(5, int(width * .75) | 1)
    region = cv2.dilate(mask, np.ones((padding, padding), np.uint8))

    # --- Fret wires and nut: straight lines crossing the board. ---
    frets = _wire_candidates(maps, 1, threshold, neck, axis, width, region)
    nuts = _wire_candidates(maps, 2, threshold, neck, axis, width, region)

    # A nut must sit near one end of the board; this avoids mistaking a capo or
    # a bright fret in the middle for the nut.
    projections = neck @ axis
    lo, hi = projections.min(), projections.max()
    nuts = [n for n in nuts if min(abs(n[1] - lo), abs(n[1] - hi)) < length * .15]
    nut = max(nuts, key=lambda item: item[0])[2] if nuts else None

    # Drop fret candidates that are really the nut detected a second time.
    fret_segments = [f[2] for f in frets
                     if nut is None or abs(f[1] - nut.mean(axis=0) @ axis) > max(3, width * .12)]
    confidence = float(maps[..., 0][mask > 0].mean())
    if not fret_segments:
        return None

    observation = Observation(neck, fret_segments, nut, confidence)
    assign_numbers(observation)
    return observation


def _wire_candidates(maps, channel, threshold, neck, axis, width, region):
    """Find straight wires in one heatmap channel that run across the board.

    Returns (confidence, position along the board axis, 2x2 segment) tuples,
    sorted from one end of the board to the other.
    """
    evidence = ((maps[..., channel] > threshold) & (region > 0)).astype(np.uint8) * 255

    # Hough describes a line by the angle of its normal. Fret wires run across
    # the neck, so their normal points *along* the board axis: only search
    # within +/-30 degrees of it. Unconstrained, Hough latches onto dense wire
    # pixels as fake lengthwise lines before it finds the real crosswise ones.
    angle = float(np.arctan2(axis[1], axis[0]) % np.pi)
    half = np.deg2rad(30)
    lo, hi = angle - half, angle + half
    # Hough angles live in [0, pi), so a window past either end wraps around.
    if lo < 0:
        ranges = [(0, hi), (np.pi + lo, np.pi)]
    elif hi > np.pi:
        ranges = [(lo, np.pi), (0, hi - np.pi)]
    else:
        ranges = [(lo, hi)]
    raw = []
    for lo, hi in ranges:
        found = cv2.HoughLines(evidence, 1, np.pi / 720, max(5, int(width * .22)),
                               min_theta=lo, max_theta=hi)
        if found is not None:
            raw.extend(found[:250, 0])
    if not raw:
        return []

    # Score every Hough proposal at once (vectorised; a Python loop per line
    # would be far too slow per frame). Each proposal is an infinite line
    # (rho, theta). Sample points along it and measure how much of it lies on
    # the board and is lit up in the heatmap.
    proposals = np.asarray(raw)
    normals = np.stack((np.cos(proposals[:, 1]), np.sin(proposals[:, 1])), axis=1)
    directions = np.stack((-normals[:, 1], normals[:, 0]), axis=1)
    # Anchor each line at its point closest to the neck centre, then sample
    # two board-widths either side of that.
    center = neck.mean(0).astype(float)
    centers = center + (proposals[:, 0] - normals @ center)[:, None] * normals
    offsets = np.linspace(-2 * width, 2 * width, max(20, int(8 * width)))
    samples = centers[:, None, :] + offsets[None, :, None] * directions[:, None, :]

    height, image_width = maps.shape[:2]
    xy = np.rint(samples).astype(int)
    in_image = ((xy[..., 0] >= 0) & (xy[..., 0] < image_width)
                & (xy[..., 1] >= 0) & (xy[..., 1] < height))
    x = np.clip(xy[..., 0], 0, image_width - 1)
    y = np.clip(xy[..., 1], 0, height - 1)
    inside = in_image & (region[y, x] > 0)       # Samples that land on the board.
    counts = inside.sum(axis=1)
    support = maps[y, x, channel]
    supported = (support > threshold) & inside  # Samples the heatmap agrees with.
    coverage = supported.sum(axis=1) / np.maximum(counts, 1)
    # Keep lines that cross enough of the board and are at least 25% lit.
    eligible = np.flatnonzero((counts >= max(4, width * .7)) & (coverage >= .25))

    candidates = []
    if len(eligible):
        # A wire's endpoints are its first and last supported sample.
        first = supported[eligible].argmax(axis=1)
        last = supported.shape[1] - 1 - supported[eligible, ::-1].argmax(axis=1)
        segments = np.stack((samples[eligible, first], samples[eligible, last]),
                            axis=1).astype(np.float32)
        confidence = (support * inside).sum(axis=1) / np.maximum(counts, 1)
        positions = segments.mean(axis=1) @ axis
        candidates = [(float(confidence[k]), float(pos), segment)
                      for k, pos, segment in zip(eligible, positions, segments)]

    # Non-maximum suppression: Hough returns many near-copies of each wire.
    # Keep the most confident one and drop anything within a few pixels of it.
    selected = []
    for candidate in sorted(candidates, key=lambda item: -item[0]):
        if all(abs(candidate[1] - other[1]) > max(2, width * .035) for other in selected):
            selected.append(candidate)
    return sorted(selected, key=lambda item: item[1])


def lattice_fit(distances, max_fret=24):
    """Work out fret numbers from pixel distances to the nut.

    On a real guitar fret n sits at ``1 - 2**(-n/12)`` of the scale length from
    the nut. A camera sees that through perspective, so in pixels:

        distance = a * ratio / (1 + c * ratio)

    Two detections with guessed fret numbers are enough to solve for ``a``
    (scale) and ``c`` (perspective). We try every guess, predict the fret
    number of every other detection, and keep the guess that lands the most
    detections on whole numbers. Missing frets are fine; detections do not
    have to be consecutive.

    Returns None when too few detections fit, or when the runner-up guess
    scores almost as well (ambiguous: better to say "unknown" than be wrong).
    """
    dist = np.asarray(distances, float)
    if len(dist) < 4 or not np.isfinite(dist).all() or np.any(dist <= 0):
        return None

    # Only use up to 5 evenly spread detections as the pair that defines a guess.
    selected = np.unique(np.linspace(0, len(dist) - 1, min(5, len(dist))).astype(int))
    # Every possible pair of fret numbers (n1 < n2) those two detections could be.
    n1, n2 = np.array(list(combinations(range(1, max_fret + 1), 2))).T
    ratio1, ratio2 = 1 - 2. ** (-n1 / 12), 1 - 2. ** (-n2 / 12)

    candidates = {}  # assignment tuple -> (score, a, c)
    for i, j in combinations(selected, 2):
        if dist[j] - dist[i] < 1e-6:
            continue
        # Solve the projective model for (a, c), for all fret-number pairs at once.
        c = (dist[j] / ratio2 - dist[i] / ratio1) / (dist[i] - dist[j])
        a = dist[i] / ratio1 + dist[i] * c
        # Discard physically impossible fits (negative scale, or a model that
        # blows up before the last fret).
        good = (a > 0) & (1 + c * (1 - 2. ** (-max_fret / 12)) > .05)
        a, c = a[good], c[good]
        if not len(a):
            continue

        # Invert the model for every detection: pixels -> ratio -> fret number.
        denom = a[:, None] - c[:, None] * dist
        ratio = np.divide(dist, denom, out=np.full_like(denom, np.nan), where=denom > 0)
        with np.errstate(invalid='ignore', divide='ignore'):
            floating = -12 * np.log2(1 - ratio)
        integer = np.rint(np.nan_to_num(floating, nan=-100, posinf=-100, neginf=-100)).astype(int)
        error = np.abs(floating - integer)
        # A detection "fits" when it lands close to a real fret number.
        inliers = (integer >= 1) & (integer <= max_fret) & (error < .18)

        # Cheaply reject guesses that fit fewer than 65% of detections.
        counts = inliers.sum(axis=1)
        eligible = np.flatnonzero(counts >= max(4, int(np.ceil(.65 * len(dist)))))
        for k in eligible:
            valid = inliers[k]
            if len(set(integer[k, valid])) != int(valid.sum()):
                continue  # Two detections claimed the same fret.
            assignment = tuple(int(n) if ok else 0 for n, ok in zip(integer[k], valid))
            # More inliers is better; smaller rounding error breaks ties.
            score = float(valid.sum() - .5 * np.mean(error[k, valid]))
            if assignment not in candidates or score > candidates[assignment][0]:
                candidates[assignment] = (score, float(a[k]), float(c[k]))

    ranked = sorted(candidates.items(), key=lambda item: -item[1][0])
    if not ranked:
        return None
    if len(ranked) > 1 and ranked[0][1][0] - ranked[1][1][0] < .025:
        return None  # Too close to call.
    assignment, (_, a, c) = ranked[0]
    return {'numbers': [n or None for n in assignment], 'a': a, 'c': c}


def assign_numbers(observation):
    """Fill in ``observation.numbers`` using the nut and fret spacing."""
    observation.numbers = [None] * len(observation.frets)
    observation.numbering = 'unknown'
    observation.estimates = []
    # Absolute numbering needs the nut as fret 0, plus enough wires to fit.
    if observation.nut is None or len(observation.frets) < 4:
        return

    # Measure each wire's distance from the nut along the neck direction.
    centers = np.array([f.mean(axis=0) for f in observation.frets])
    origin = observation.nut.mean(axis=0)
    _, _, vectors = np.linalg.svd(centers - origin, full_matrices=False)
    axis = vectors[0]
    if np.median((centers - origin) @ axis) < 0:
        axis = -axis  # Point the axis away from the nut.
    distances = (centers - origin) @ axis
    order = np.argsort(distances)

    fit = lattice_fit(distances[order])
    if fit is None:
        return
    for i, n in zip(order, fit['numbers']):
        observation.numbers[i] = n
    observation.numbering = 'nut-anchored fit'
    estimate_missing_frets(observation, fit, origin, axis)


def estimate_missing_frets(observation, fit, origin, axis, max_missing=4):
    """Fill short gaps between numbered frets using the accepted spacing fit.

    Estimates are stored separately from ``frets``/``numbers`` so they can never
    be mistaken for measurements or feed into the next frame's numbering fit.
    """
    observation.estimates = []
    if observation.nut is None or observation.numbering != 'nut-anchored fit':
        return
    measured = [(n, line) for n, line in zip(observation.numbers, observation.frets)
                if n is not None]
    if len(measured) < 4 or len({n for n, _ in measured}) != len(measured):
        return
    # Walk consecutive known wires (the nut counts as fret 0).
    anchors = [(0, observation.nut)] + sorted(measured, key=lambda item: item[0])

    def position(n):
        """Predicted pixel distance of fret n from the nut, from the fit."""
        ratio = 1 - 2. ** (-n / 12)
        denominator = 1 + fit['c'] * ratio
        return fit['a'] * ratio / denominator if denominator > .05 else float('nan')

    for (low, left), (high, right) in zip(anchors, anchors[1:]):
        # Only fill small gaps (1..max_missing frets missing).
        if not 1 <= high - low - 1 <= max_missing:
            continue
        sl, sh = position(low), position(high)
        if not np.isfinite([sl, sh]).all() or sh - sl < 2 * (high - low):
            continue
        # Endpoints have no fixed order; line them up before interpolating.
        left, right = np.asarray(left, float), np.asarray(right, float)
        if np.linalg.norm(left - right[::-1]) < np.linalg.norm(left - right):
            right = right[::-1]
        if (right.mean(0) - left.mean(0)) @ axis <= 0:
            continue
        # Skip the gap if the two anchor wires don't sit where the fit predicts.
        spacing = min(position(low + 1) - sl, sh - position(high - 1))
        residual = max(abs((left.mean(0) - origin) @ axis - sl),
                       abs((right.mean(0) - origin) @ axis - sh))
        if spacing <= 0 or residual > .2 * spacing:
            continue

        for number in range(low + 1, high):
            # Blend the two anchor wires, weighted by where fret `number` falls.
            weight = (position(number) - sl) / (sh - sl)
            if not 0 < weight < 1:
                continue
            line = ((1 - weight) * left + weight * right).astype(np.float32)
            if not np.isfinite(line).all() or np.linalg.norm(line[1] - line[0]) < 4:
                continue
            # Reject estimates that fall off the detected neck.
            samples = line[0] + np.linspace(0, 1, 5)[:, None] * (line[1] - line[0])
            if any(cv2.pointPolygonTest(observation.neck.astype(np.float32),
                                        (float(p[0]), float(p[1])), True) < -2 for p in samples):
                continue
            # A real but unnumbered detection may already occupy this slot.
            center = line.mean(0)
            if any(np.linalg.norm(center - f.mean(0)) < max(2, .25 * spacing)
                   for f in observation.frets):
                continue
            observation.estimates.append(EstimatedFret(line, number))


def retain_missing_frets(detected, warped, max_missing=4):
    """Carry briefly-missing frets forward from the previous frame.

    ``warped`` is last frame's geometry moved by the validated camera motion.
    Any numbered fret it has that this frame is missing (inside a small gap)
    is added as a 'tracked' estimate.
    """
    numbered = [(n, line) for n, line in zip(detected.numbers, detected.frets)
                if n is not None]
    if len(numbered) < 2:
        return
    low = min(n for n, _ in numbered)
    if warped.nut is not None:
        low = 0
    # Which fret numbers fall inside a gap small enough to fill?
    present = {n for n, _ in numbered}
    boundaries = sorted(present | {low})
    permitted = {n for left, right in zip(boundaries, boundaries[1:])
                 if right - left - 1 <= max_missing for n in range(left + 1, right)}

    candidates = [EstimatedFret(line, n, 'tracked') for n, line in
                  zip(warped.numbers, warped.frets) if n is not None]
    candidates += [EstimatedFret(e.line, e.number, 'tracked') for e in warped.estimates]
    width = min(cv2.minAreaRect(detected.neck)[1])
    for estimate in candidates:
        if estimate.number not in permitted or estimate.number in present:
            continue
        # Don't duplicate a wire that was actually detected here.
        if any(np.linalg.norm(estimate.line.mean(0) - f.mean(0)) < max(2, width * .15)
               for f in detected.frets):
            continue
        # Must lie on the current neck.
        if any(cv2.pointPolygonTest(detected.neck.astype(np.float32),
                                    (float(p[0]), float(p[1])), True) < -2
               for p in estimate.line):
            continue
        detected.estimates.append(estimate)
        present.add(estimate.number)


def transform_observation(observation, matrix):
    """Apply a 3x3 homography to every point in an observation (returns a copy)."""
    def transform(points):
        points = np.asarray(points, np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(points, matrix).reshape(-1, 2)

    return Observation(
        transform(observation.neck),
        [transform(f) for f in observation.frets],
        transform(observation.nut) if observation.nut is not None else None,
        observation.confidence,
        list(observation.numbers),
        observation.numbering,
        [EstimatedFret(transform(e.line), e.number, e.method) for e in observation.estimates])


def _nearest_unmatched(line, others, used, radius):
    """Index of the closest line in ``others`` not in ``used``, or None if none is within ``radius``."""
    distances = [np.linalg.norm(line.mean(0) - other.mean(0)) if j not in used else np.inf
                 for j, other in enumerate(others)]
    j = int(np.argmin(np.asarray(distances, dtype=float)))
    return j if distances[j] < radius else None


class FretboardTracker:
    """Stabilises fretboard geometry across frames.

    States:
      'detected' - the model found the board this frame.
      'tracked'  - the model missed it, but optical flow moved last frame's board.
      'lost'     - neither; nothing is shown.

    Tracking is only trusted for ``max_gap`` seconds, so a stale overlay can't
    freeze on screen.
    """

    def __init__(self, max_gap=.5, smoothing=.06):
        if not np.isfinite([max_gap, smoothing]).all() or max_gap <= 0 or smoothing < 0:
            raise ValueError('Invalid tracker timing')
        self.max_gap = max_gap      # Seconds a board may go unseen before we drop it.
        self.smoothing = smoothing  # Time constant (s) for blending wire positions.
        self.reset()

    def reset(self):
        self.previous_gray = None         # Last frame in grayscale, for optical flow.
        self.observation = None           # Current best geometry.
        self.last_time = None             # Timestamp of the last update.
        self.last_detection = None        # Last time frets were *numbered* by a nut fit.
        self.state = 'lost'
        self.flow_inliers = 0             # Diagnostic: points agreeing on the motion.
        self.last_board_detection = None  # Last time the model saw the board at all.

    def _motion(self, gray):
        """Move last frame's observation to this frame using optical flow.

        Returns the moved observation, or None if the motion is not trustworthy.
        """
        if self.observation is None or self.previous_gray is None:
            return None

        # Pick trackable corners on the neck in the previous frame.
        mask = np.zeros_like(self.previous_gray)
        cv2.fillConvexPoly(mask, self.observation.neck.astype(np.int32), 255)
        points = cv2.goodFeaturesToTrack(self.previous_gray, 150, .01, 5, mask=mask)
        if points is None or len(points) < 8:
            return None

        # Track them forward, then backward; keep points that return to where
        # they started (forward-backward consistency check).
        current, status, _ = cv2.calcOpticalFlowPyrLK(self.previous_gray, gray, points, points.copy())
        if current is None:
            return None
        back, status_back, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous_gray, current, current.copy())
        if back is None:
            return None
        good = ((status.ravel() > 0) & (status_back.ravel() > 0)
                & (np.linalg.norm(points - back, axis=2).ravel() < 1.5))
        p, q = points[good], current[good]
        if len(p) < 8:
            return None

        # Fit one homography to the motion with RANSAC to ignore outliers.
        diag = np.hypot(*gray.shape)
        matrix, inliers = cv2.findHomography(p, q, cv2.RANSAC, max(2, diag * .002))
        if matrix is None or inliers is None:
            return None
        inside = inliers.ravel() > 0
        self.flow_inliers = int(inside.sum())
        if self.flow_inliers < 8 or inside.mean() < .6:
            return None

        # Reject motion measured only on a hand or a tiny patch of the board.
        source = p[inside].reshape(-1, 2)
        source_area = cv2.contourArea(cv2.convexHull(source))
        board_area = cv2.contourArea(self.observation.neck)
        if source_area < board_area * .15:
            return None

        # Sanity-check the result: the board shouldn't suddenly halve/double in
        # size or jump across the frame.
        warped = transform_observation(self.observation, matrix)
        new_area = cv2.contourArea(warped.neck)
        if not np.isfinite(warped.neck).all() or not .5 < new_area / max(board_area, 1) < 2:
            return None
        if np.linalg.norm(warped.neck.mean(0) - self.observation.neck.mean(0)) > diag * .25:
            return None
        return warped

    def update(self, frame, detected, timestamp):
        """Combine this frame's detection (may be None) with tracking history."""
        if not np.isfinite(timestamp):
            raise ValueError('Timestamp must be finite')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Start over if time went backwards, jumped too far, or the frame size changed.
        if self.last_time is not None and (
                timestamp <= self.last_time
                or timestamp - self.last_time > self.max_gap
                or self.previous_gray is None
                or gray.shape != self.previous_gray.shape):
            self.reset()

        # Where last frame's board should be now, according to camera motion.
        warped = None
        self.flow_inliers = 0
        if self.observation is not None and self.previous_gray is not None:
            warped = self._motion(gray)

        if detected is not None:
            if warped is not None:
                # Smooth each detected wire toward its matching tracked wire.
                # Only individually matched wires are blended, never whole stale rails.
                dt = timestamp - self.last_time
                alpha = 1 - np.exp(-dt / max(self.smoothing, 1e-6))
                width = min(cv2.minAreaRect(detected.neck)[1])
                radius = max(2, width * .15)
                used = set()
                for i, line in enumerate(detected.frets):
                    if not warped.frets:
                        break
                    j = _nearest_unmatched(line, warped.frets, used, radius)
                    if j is not None:
                        used.add(j)
                        old = warped.frets[j]
                        if np.linalg.norm(line - old[::-1]) < np.linalg.norm(line - old):
                            old = old[::-1]  # Match endpoint order before blending.
                        detected.frets[i] = (alpha * line + (1 - alpha) * old).astype(np.float32)

                # This frame couldn't number its frets (e.g. nut hidden), but a
                # recent frame could: copy numbers across from matched wires.
                if (detected.numbering == 'unknown' and self.last_detection is not None
                        and timestamp - self.last_detection < self.max_gap):
                    numbered_used = set()
                    for i, line in enumerate(detected.frets):
                        if not warped.frets:
                            break
                        j = _nearest_unmatched(line, warped.frets, numbered_used, radius)
                        if j is not None:
                            detected.numbers[i] = warped.numbers[j]
                            numbered_used.add(j)
                    if any(n is not None for n in detected.numbers):
                        detected.numbering = 'short-term tracked'
                        retain_missing_frets(detected, warped)

            self.observation = detected
            if detected.numbering == 'nut-anchored fit':
                self.last_detection = timestamp
            self.state = 'detected'
            self.last_board_detection = timestamp

        elif (warped is not None and self.last_board_detection is not None
              and timestamp - self.last_board_detection <= self.max_gap):
            # No detection, but motion is reliable: coast on the tracked board
            # and let its confidence decay the longer we go without seeing it.
            self.observation = warped
            self.state = 'tracked'
            for estimate in self.observation.estimates:
                estimate.method = 'tracked'
            self.observation.confidence *= np.exp(-(timestamp - self.last_time) / self.max_gap)

        else:
            self.observation = None
            self.state = 'lost'
            self.last_detection = None

        # Board position and fret *numbering* expire independently. A recent
        # unnumbered detection must not keep old fret numbers alive through a
        # later motion-only gap.
        if (self.observation is not None and self.last_detection is not None
                and timestamp - self.last_detection >= self.max_gap):
            self.observation.numbers = [None] * len(self.observation.frets)
            self.observation.numbering = 'unknown'
            self.observation.estimates = []

        self.previous_gray = gray
        self.last_time = timestamp
        return self.observation

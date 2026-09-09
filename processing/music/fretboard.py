"""Continuous (fret, string) board coordinates from detected fretboard geometry.

The dense model has no string head, so string positions are **interpolated**
across the detected board, never measured. Fret positions are measured.

Board space is (ratio, across):

* ``ratio`` is the fraction of scale length from the nut, ``1 - 2**(-n/12)`` at
  fret ``n``. That is proportional to physical distance along the neck, so the
  image-to-board map is an ordinary homography. Fret *number* is not, which is
  why numbering is undone only after the projective step.
* ``across`` runs 0..1 between the two ends of a fret wire. Which end carries
  string 1 cannot be recovered from geometry alone; see ``flipped``.
"""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np

# Strings sit inside the edges of the board. This is an assumption about
# instrument setup, not a measurement, and it directly scales string error.
DEFAULT_STRING_INSET = .09
MIN_ANCHORS = 3


def fret_ratio(n):
    """Fraction of scale length from the nut to fret ``n``."""
    return 1 - 2. ** (-np.asarray(n, float) / 12)


def ratio_to_fret(ratio):
    """Invert ``fret_ratio``. Positions at or beyond the bridge become inf."""
    ratio = np.asarray(ratio, float)
    remainder = 1 - ratio
    with np.errstate(divide='ignore', invalid='ignore'):
        fret = -12 * np.log2(remainder)
    return np.where(remainder > 0, fret, np.inf)


def _fit_edge(points):
    """Robust straight-line fit, returned as (point on line, unit direction)."""
    vx, vy, x0, y0 = cv2.fitLine(np.asarray(points, np.float32), cv2.DIST_HUBER, 0, .01, .01).ravel()
    return np.array([x0, y0]), np.array([vx, vy])


def _intersect(centre, direction, edge):
    """Where the line centre+t*direction meets an edge line, or None if parallel."""
    origin, along = edge
    denominator = direction[0]*along[1] - direction[1]*along[0]
    if abs(denominator) < 1e-9:
        return None
    delta = origin - centre
    distance = (delta[0]*along[1] - delta[1]*along[0]) / denominator
    return centre + distance*direction


def _point_line_distance(points, edge):
    origin, along = edge
    delta = np.asarray(points, float) - origin
    return np.abs(delta[:, 0]*along[1] - delta[:, 1]*along[0])


def _endpoint_edge(points, across, side, width):
    """Consensus outer rail; require three endpoints, tolerate inward truncation.

    Outer outliers are penalized, rather than blindly taking a convex hull.
    If all wires end short consistently, their true width is unknowable here.
    """
    tolerance = max(1., width * .035)
    best = None
    best_score = -float('inf')
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            delta = points[j] - points[i]
            length = np.linalg.norm(delta)
            if length < width * .25:
                continue
            direction = delta / length
            normal = np.array([-direction[1], direction[0]])
            if normal @ across < 0:
                normal = -normal
            signed = (points - points[i]) @ normal
            support = np.abs(signed) <= tolerance
            outside = signed < -tolerance if side == 0 else signed > tolerance
            score = support.sum() - 2 * outside.sum()
            if support.sum() >= MIN_ANCHORS and score > best_score:
                best_score = score
                best = _fit_edge(points[support])
    return best


@dataclass
class BoardTransform:
    """Projective map between image pixels and board coordinates."""
    matrix: np.ndarray
    inverse: np.ndarray
    strings: int = 6
    inset: float = DEFAULT_STRING_INSET
    flipped: bool = False
    anchors: int = 0
    residual: float = 0.
    edge_residual: float = 0.

    def to_board(self, points):
        points = np.asarray(points, np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(points, self.matrix.astype(np.float32)).reshape(-1, 2)

    def to_image(self, board):
        board = np.asarray(board, np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(board, self.inverse.astype(np.float32)).reshape(-1, 2)

    def locate(self, points):
        """Return continuous (fret, string) for image points.

        Fret is the equal-tempered fret coordinate: a fingertip at 4.3 sits
        between wires 4 and 5 and therefore sounds fret 5. String is a
        continuous index where 1.0 is string 1; values outside 1..strings mean
        the point is off the string band.
        """
        board = self.to_board(points)
        fret = ratio_to_fret(board[:, 0])
        span = 1 - 2 * self.inset
        string = (1 + (board[:, 1] - .5) / span if self.strings == 1 else
                  1 + (board[:, 1] - self.inset) / span * (self.strings - 1))
        return fret, string

    def board_point(self, string, fret, depth=.5):
        """Image point for a fretted contact, ``depth`` of the way into the cell."""
        low, high = fret_ratio(max(fret - 1, 0)), fret_ratio(fret)
        ratio = float(low + (high - low) * depth) if fret >= 1 else float(fret_ratio(0))
        across = (.5 if self.strings == 1 else
                  self.inset + (string - 1) / (self.strings - 1) * (1 - 2 * self.inset))
        return self.to_image([[ratio, across]])[0]

    def string_polyline(self, string, max_fret=22, samples=24):
        frets = np.linspace(0, max_fret, samples)
        across = (.5 if self.strings == 1 else
                  self.inset + (string - 1) / (self.strings - 1) * (1 - 2 * self.inset))
        return self.to_image(np.stack([fret_ratio(frets), np.full(samples, across)], axis=1))

    def cell_radius(self, string, fret):
        """Half the smaller of the local fret spacing and string spacing, in pixels.

        Used as the tolerance for calling a fingertip 'on' a cell, so the
        tolerance shrinks correctly high up the neck and with distance.
        """
        centre = self.board_point(string, fret)
        along = np.linalg.norm(self.board_point(string, fret, 1.) - self.board_point(string, fret, 0.))
        neighbour = min(string + 1, self.strings) if self.strings > 1 else string
        if neighbour == string:
            neighbour = max(string - 1, 1)
        across = np.linalg.norm(self.board_point(neighbour, fret) - centre)
        candidates = [v for v in (along, across) if np.isfinite(v) and v > 0]
        return float(min(candidates) / 2) if candidates else 0.


def board_transform(observation, strings=6, inset=DEFAULT_STRING_INSET, flipped=False,
                    max_residual=.05, max_edge_residual=.06):
    """Fit a board transform from numbered wires. Returns None without enough evidence.

    Only measured wires and the nut are used. Interpolated estimates are
    consistent with the numbering fit by construction, so including them would
    add apparent support without adding evidence.
    """
    if not 1 <= strings <= 12 or not 0 <= inset < .5:
        raise ValueError('Invalid string count or inset')
    anchors = {}
    if observation.nut is not None:
        anchors[0] = np.asarray(observation.nut, float)
    for number, line in zip(observation.numbers, observation.frets):
        if number is not None:
            anchors.setdefault(int(number), np.asarray(line, float))
    numbers = sorted(anchors)
    if len(numbers) < MIN_ANCHORS:
        return None
    raw = np.array([anchors[n] for n in numbers])
    if not np.isfinite(raw).all():
        return None
    lines = raw
    centres = lines.mean(axis=1)
    _, _, vectors = np.linalg.svd(centres - centres.mean(0), full_matrices=False)
    along = vectors[0]
    # SVD axes have arbitrary signs. Orient by increasing fret number so small
    # camera movements cannot silently reverse the string order between frames.
    if along @ (centres[-1] - centres[0]) < 0:
        along = -along
    across = np.array([-along[1], along[0]])
    # Wire endpoints carry no string identity; impose one consistent order.
    order = np.argsort(lines @ across, axis=1)
    ordered = np.take_along_axis(lines, order[..., None], axis=1)
    width = float(np.median(np.linalg.norm(ordered[:, 1]-ordered[:, 0], axis=1)))
    if width <= 0:
        return None
    # Short/occluded wires may lie inside the rails but must not shrink them.
    edges = tuple(_endpoint_edge(ordered[:, side], across, side, width)
                  for side in (0, 1))
    if any(edge is None for edge in edges):
        return None
    edge_residual = float(max(np.median(np.sort(
        _point_line_distance(ordered[:, side], edges[side]))[:MIN_ANCHORS])
        for side in (0, 1)) / width)
    refined, kept = [], []
    for number, line in zip(numbers, lines):
        centre, direction = line.mean(0), line[1]-line[0]
        span = [_intersect(centre, direction, edge) for edge in edges]
        if any(point is None for point in span):
            continue
        span = np.asarray(span, dtype=float)
        if not np.isfinite(span).all():
            continue
        refined.append(span)
        kept.append(number)
    if len(kept) < MIN_ANCHORS:
        return None
    numbers = kept
    source = np.array(refined, np.float32).reshape(-1, 2)
    ratios = np.repeat(fret_ratio(np.array(numbers, float)), 2)
    ends = np.tile([0., 1.], len(numbers))
    target = np.stack([ratios, 1 - ends if flipped else ends], axis=1).astype(np.float32)
    matrix, _ = cv2.findHomography(source, target, 0)
    if matrix is None or not np.isfinite(matrix).all():
        return None
    ok, inverse = cv2.invert(matrix)
    if not ok or not np.isfinite(inverse).all():
        return None
    reprojected = cv2.perspectiveTransform(source.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    residual = float(np.abs(reprojected - target).max())
    if not np.isfinite(residual) or residual > max_residual:
        return None
    if not np.isfinite(edge_residual) or edge_residual > max_edge_residual:
        return None
    return BoardTransform(matrix, inverse, strings, inset, flipped, len(numbers),
                          residual, edge_residual)

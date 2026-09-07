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
DEFAULT_STRING_INSET = .12
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


def clip_to_hull(centre, direction, hull):
    """Intersect an infinite line with a convex polygon (Cyrus-Beck).

    Detected wire endpoints stop wherever the evidence stops, so they are not
    comparable between wires. Clipping every wire to the same neck boundary is
    what makes the across coordinate mean the same thing on each of them.
    """
    direction = np.asarray(direction, float)
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        return None
    direction = direction / norm
    centre = np.asarray(centre, float)
    hull = np.asarray(hull, float).reshape(-1, 2)
    if len(hull) < 3:
        return None
    area = .5 * np.sum(hull[:, 0]*np.roll(hull[:, 1], -1) - np.roll(hull[:, 0], -1)*hull[:, 1])
    polygon = hull if area > 0 else hull[::-1]
    low, high = -np.inf, np.inf
    for index in range(len(polygon)):
        start, end = polygon[index], polygon[(index+1) % len(polygon)]
        edge = end - start
        normal = np.array([edge[1], -edge[0]])
        denominator = normal @ direction
        numerator = normal @ (start - centre)
        if abs(denominator) < 1e-12:
            if numerator < 0:
                return None
            continue
        distance = numerator / denominator
        if denominator > 0:
            high = min(high, distance)
        else:
            low = max(low, distance)
    if not np.isfinite(low) or not np.isfinite(high) or high - low < 1e-6:
        return None
    return np.array([centre + low*direction, centre + high*direction])


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
        string = 1 + (board[:, 1] - self.inset) / span * (self.strings - 1)
        return fret, string

    def board_point(self, string, fret, depth=.5):
        """Image point for a fretted contact, ``depth`` of the way into the cell."""
        low, high = fret_ratio(max(fret - 1, 0)), fret_ratio(fret)
        ratio = float(low + (high - low) * depth) if fret >= 1 else float(fret_ratio(0))
        across = self.inset + (string - 1) / max(self.strings - 1, 1) * (1 - 2 * self.inset)
        return self.to_image([[ratio, across]])[0]

    def string_polyline(self, string, max_fret=22, samples=24):
        frets = np.linspace(0, max_fret, samples)
        across = self.inset + (string - 1) / max(self.strings - 1, 1) * (1 - 2 * self.inset)
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
    if not 2 <= strings <= 12 or not 0 <= inset < .5:
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
    # Re-cut every wire against the neck boundary so both ends mean 'board edge'.
    clipped, kept = [], []
    for number, line in zip(numbers, raw):
        span = clip_to_hull(line.mean(0), line[1]-line[0], observation.neck)
        if span is not None:
            clipped.append(span)
            kept.append(number)
    if len(kept) < MIN_ANCHORS:
        return None
    numbers, lines = kept, np.array(clipped)
    centres = lines.mean(axis=1)
    _, _, vectors = np.linalg.svd(centres - centres.mean(0), full_matrices=False)
    along = vectors[0]
    across = np.array([-along[1], along[0]])
    # Wire endpoints carry no string identity; impose one consistent order.
    order = np.argsort(lines @ across, axis=1)
    ordered = np.take_along_axis(lines, order[..., None], axis=1)
    # A convex hull of a segmentation mask has rounded ends, so individual chords
    # disagree about where the board edge is. Fit one straight edge per side from
    # all of them, then take each wire's span between those two edges.
    edges = (_fit_edge(ordered[:, 0]), _fit_edge(ordered[:, 1]))
    width = float(np.median(np.linalg.norm(ordered[:, 1]-ordered[:, 0], axis=1)))
    if width <= 0:
        return None
    edge_residual = float(max(np.median(_point_line_distance(ordered[:, side], edges[side]))
                              for side in (0, 1)) / width)
    refined, kept = [], []
    for number, line in zip(numbers, lines):
        centre, direction = line.mean(0), line[1]-line[0]
        span = [_intersect(centre, direction, edge) for edge in edges]
        if any(point is None for point in span) or not np.isfinite(span).all():
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

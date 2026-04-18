"""Geometry helper functions for vision pipeline.

This module intentionally keeps pure functions only. It provides low-level
geometric utilities used by tracker and ROI rule engines.
"""

from __future__ import annotations

from typing import Sequence


def _normalize_bbox(bbox: Sequence[float] | Sequence[int]) -> tuple[float, float, float, float] | None:
    """Validate and normalize bbox input to (x1, y1, x2, y2).

    Returns:
        Normalized tuple when valid, otherwise None.
    """
    if bbox is None or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    return x1, y1, x2, y2


def bbox_center(bbox: Sequence[float] | Sequence[int]) -> tuple[float, float]:
    """Compute bbox center point.

    Args:
        bbox: [x1, y1, x2, y2]

    Returns:
        (cx, cy). Returns (0.0, 0.0) when input is invalid.
    """
    normalized = _normalize_bbox(bbox)
    if normalized is None:
        return 0.0, 0.0

    x1, y1, x2, y2 = normalized
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def point_in_polygon(point: Sequence[float] | Sequence[int], polygon: Sequence[Sequence[float] | Sequence[int]]) -> bool:
    """Check whether a point is inside or on the boundary of a polygon.

    Uses ray-casting with explicit boundary check for stable behavior.
    """
    if point is None or len(point) != 2:
        return False
    if polygon is None or len(polygon) < 3:
        return False

    try:
        px, py = float(point[0]), float(point[1])
        pts = [(float(p[0]), float(p[1])) for p in polygon if len(p) >= 2]
    except (TypeError, ValueError):
        return False

    if len(pts) < 3:
        return False

    eps = 1e-9

    # Boundary check: treat points on edges as inside.
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]

        cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
        if abs(cross) > eps:
            continue

        min_x = min(x1, x2) - eps
        max_x = max(x1, x2) + eps
        min_y = min(y1, y2) - eps
        max_y = max(y1, y2) + eps
        if min_x <= px <= max_x and min_y <= py <= max_y:
            return True

    inside = False
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]

        intersects = (y1 > py) != (y2 > py)
        if not intersects:
            continue

        x_at_y = x1 + (py - y1) * (x2 - x1) / ((y2 - y1) + eps)
        if px < x_at_y:
            inside = not inside

    return inside


def bbox_in_zone(bbox: Sequence[float] | Sequence[int], polygon: Sequence[Sequence[float] | Sequence[int]]) -> bool:
    """Check whether bbox is in zone based on bbox center point."""
    center = bbox_center(bbox)
    return point_in_polygon(center, polygon)


def clip_bbox_to_image(
    bbox: Sequence[float] | Sequence[int], width: int, height: int
) -> list[int]:
    """Clip bbox into image range and return integer bbox.

    Args:
        bbox: [x1, y1, x2, y2]
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        [x1, y1, x2, y2] clipped to valid range.
        Returns [0, 0, 0, 0] when inputs are invalid.
    """
    normalized = _normalize_bbox(bbox)
    if normalized is None:
        return [0, 0, 0, 0]
    if width <= 0 or height <= 0:
        return [0, 0, 0, 0]

    x1, y1, x2, y2 = normalized

    max_x = width - 1
    max_y = height - 1

    x1_i = int(min(max(x1, 0.0), max_x))
    y1_i = int(min(max(y1, 0.0), max_y))
    x2_i = int(min(max(x2, 0.0), max_x))
    y2_i = int(min(max(y2, 0.0), max_y))

    if x2_i < x1_i:
        x1_i, x2_i = x2_i, x1_i
    if y2_i < y1_i:
        y1_i, y2_i = y2_i, y1_i

    return [x1_i, y1_i, x2_i, y2_i]


def compute_iou(
    box_a: Sequence[float] | Sequence[int], box_b: Sequence[float] | Sequence[int]
) -> float:
    """Compute IoU (Intersection over Union) between two bboxes."""
    a = _normalize_bbox(box_a)
    b = _normalize_bbox(box_b)
    if a is None or b is None:
        return 0.0

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    if ax2 < ax1:
        ax1, ax2 = ax2, ax1
    if ay2 < ay1:
        ay1, ay2 = ay2, ay1
    if bx2 < bx1:
        bx1, bx2 = bx2, bx1
    if by2 < by1:
        by1, by2 = by2, by1

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return float(inter_area / union)

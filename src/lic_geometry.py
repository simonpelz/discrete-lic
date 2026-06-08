"""Voronoi geometry helpers shared between the pattern generator and the LIC engine."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from scipy.spatial import Voronoi


def voronoi_finite_polygons_2d(vor: Voronoi, radius: float | None = None):
    """Reconstruct infinite Voronoi regions to finite regions.

    Adapted from the SciPy Voronoi example.
    Returns (regions, vertices).
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2

    all_ridges: dict = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_idx in enumerate(vor.point_region):
        vertices = vor.regions[region_idx]
        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0 and v2 >= 0:
                continue
            t = vor.points[p2] - vor.points[p1]
            t_norm = np.linalg.norm(t)
            if t_norm == 0:
                continue
            t /= t_norm
            n = np.array([-t[1], t[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius
            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = [v for _, v in sorted(zip(angles, new_region))]
        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)


def clip_polygon_to_rect(
    polygon: Sequence[Tuple[float, float]],
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> List[Tuple[float, float]]:
    """Clip a polygon to an axis-aligned rectangle using Sutherland-Hodgman."""

    def clip(subject: List[Tuple[float, float]], edge_fn, intersect_fn):
        if not subject:
            return []
        output = []
        prev = subject[-1]
        prev_inside = edge_fn(prev)
        for curr in subject:
            curr_inside = edge_fn(curr)
            if curr_inside:
                if not prev_inside:
                    output.append(intersect_fn(prev, curr))
                output.append(curr)
            elif prev_inside:
                output.append(intersect_fn(prev, curr))
            prev, prev_inside = curr, curr_inside
        return output

    def intersect_vertical(p1, p2, x):
        x1, y1 = p1
        x2, y2 = p2
        if x2 == x1:
            return (x, y1)
        t = (x - x1) / (x2 - x1)
        return (x, y1 + t * (y2 - y1))

    def intersect_horizontal(p1, p2, y):
        x1, y1 = p1
        x2, y2 = p2
        if y2 == y1:
            return (x1, y)
        t = (y - y1) / (y2 - y1)
        return (x1 + t * (x2 - x1), y)

    poly = list(polygon)
    poly = clip(poly, lambda p: p[0] >= xmin, lambda a, b: intersect_vertical(a, b, xmin))
    poly = clip(poly, lambda p: p[0] <= xmax, lambda a, b: intersect_vertical(a, b, xmax))
    poly = clip(poly, lambda p: p[1] >= ymin, lambda a, b: intersect_horizontal(a, b, ymin))
    poly = clip(poly, lambda p: p[1] <= ymax, lambda a, b: intersect_horizontal(a, b, ymax))
    return poly

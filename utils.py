# utils.py
# Small helpers: timings, point conversions, normalization, color constants
import time
import numpy as np
import cv2

# Colors in BGR (OpenCV)
COLOR_INLIER = (0, 255, 0)      # green
COLOR_OUTLIER = (0, 0, 255)     # red
COLOR_MATCH_LINE = (255, 0, 0)  # blue
COLOR_TEXT = (230, 230, 230)    # near white

def tic():
    return time.time()

def toc(t0):
    return time.time() - t0

def to_homogeneous(pts):
    pts = np.asarray(pts, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    ones = np.ones((pts.shape[0], 1), dtype=pts.dtype)
    return np.hstack([pts, ones])

def from_homogeneous(pts_h):
    pts_h = np.asarray(pts_h, dtype=float)
    pts = pts_h[:, :2] / pts_h[:, 2:3]
    return pts

def normalize_points(pts):
    """
    Normalize 2D points (Nx2). Returns similarity matrix T (3x3) and normalized points (Nx2).
    Translation to centroid and scaling so that mean distance = sqrt(2).
    """
    pts = np.asarray(pts, dtype=float)
    centroid = pts.mean(axis=0)
    pts_centered = pts - centroid
    mean_dist = np.mean(np.sqrt((pts_centered ** 2).sum(axis=1)))
    if mean_dist <= 0:
        scale = 1.0
    else:
        scale = np.sqrt(2) / mean_dist
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ], dtype=float)
    pts_h = to_homogeneous(pts)
    pts_norm_h = (T @ pts_h.T).T
    pts_norm = from_homogeneous(pts_norm_h)
    return T, pts_norm

def draw_text(img, text, pos=(10,30), color=COLOR_TEXT, scale=0.7, thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

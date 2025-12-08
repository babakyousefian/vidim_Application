# model_estimation.py
import numpy as np
from utils import normalize_points, to_homogeneous, from_homogeneous
import cv2

def estimate_homography(pts1, pts2):
    """
    Normalized DLT homography. pts1, pts2 arrays of shape (N,2).
    Returns 3x3 homography or None.
    """
    pts1 = np.asarray(pts1)
    pts2 = np.asarray(pts2)
    if pts1.shape[0] < 4:
        return None
    T1, p1n = normalize_points(pts1)
    T2, p2n = normalize_points(pts2)
    A = []
    for (x, y), (xp, yp) in zip(p1n, p2n):
        A.append([-x, -y, -1, 0, 0, 0, xp * x, xp * y, xp])
        A.append([0, 0, 0, -x, -y, -1, yp * x, yp * y, yp])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1, :]
    Hn = h.reshape(3, 3)
    H = np.linalg.inv(T2) @ Hn @ T1
    if abs(H[2,2]) < 1e-12:
        H = H / (np.linalg.norm(H) + 1e-12)
    else:
        H = H / H[2,2]
    return H

def estimate_fundamental(pts1, pts2):
    """
    Normalized 8-point algorithm for fundamental matrix.
    pts1, pts2: (N,2)
    Returns 3x3 fundamental matrix (rank-2 enforced) or None
    """
    pts1 = np.asarray(pts1)
    pts2 = np.asarray(pts2)
    if pts1.shape[0] < 8:
        return None
    T1, p1n = normalize_points(pts1)
    T2, p2n = normalize_points(pts2)
    A = []
    for (x, y), (xp, yp) in zip(p1n, p2n):
        A.append([xp * x, xp * y, xp, yp * x, yp * y, yp, x, y, 1])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    f = Vt[-1, :]
    F_raw = f.reshape(3, 3)
    U, S, Vt_f = np.linalg.svd(F_raw)
    S[2] = 0
    F_rank2 = U @ np.diag(S) @ Vt_f
    F = T2.T @ F_rank2 @ T1
    # normalize scale
    F = F / (np.linalg.norm(F) + 1e-12)
    return F

def reprojection_error_homography(H, pts1, pts2):
    """
    Compute Euclidean reprojection error between H*pts1 and pts2 (per-point).
    """
    pts1h = to_homogeneous(pts1)
    proj = (H @ pts1h.T).T
    proj_xy = proj[:, :2] / proj[:, 2:3]
    err = np.linalg.norm(proj_xy - pts2, axis=1)
    return err

def sampson_error(F, pts1, pts2):
    """
    Sampson distance for fundamental matrix; vectorized.
    """
    pts1h = to_homogeneous(pts1)
    pts2h = to_homogeneous(pts2)
    # l2 = F * x1
    l2 = (F @ pts1h.T).T
    # l1 = F^T * x2
    l1 = (F.T @ pts2h.T).T
    numer = np.sum(pts2h * (F @ pts1h.T).T, axis=1)**2
    denom = l1[:,0]**2 + l1[:,1]**2 + l2[:,0]**2 + l2[:,1]**2
    denom[denom==0] = 1e-12
    return numer / denom

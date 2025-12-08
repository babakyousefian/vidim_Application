# ransac.py
import numpy as np
import random
from model_estimation import estimate_homography, estimate_fundamental, reprojection_error_homography, sampson_error

def ransac_generic(pts1, pts2, model='homography', threshold=3.0, confidence=0.99, max_iter=5000):
    """
    Classic RANSAC. Returns (best_model, inlier_mask_bool_array, iterations_done)
    pts1, pts2: Nx2 arrays
    model: 'homography' or 'fundamental'
    threshold: inlier threshold (pixels for homography, Sampson for F)
    """
    pts1 = np.asarray(pts1)
    pts2 = np.asarray(pts2)
    N = pts1.shape[0]
    if N == 0:
        return None, np.array([], dtype=bool), 0
    m = 4 if model == 'homography' else 8
    best_model = None
    best_inlier_mask = np.zeros(N, dtype=bool)
    it = 0
    k = float('inf')
    while it < k and it < max_iter:
        it += 1
        if N < m:
            break
        idx = random.sample(range(N), m)
        s1 = pts1[idx]
        s2 = pts2[idx]
        if model == 'homography':
            M = estimate_homography(s1, s2)
            if M is None:
                continue
            errs = reprojection_error_homography(M, pts1, pts2)
        else:
            F = estimate_fundamental(s1, s2)
            if F is None:
                continue
            errs = sampson_error(F, pts1, pts2)
        inlier_mask = errs < threshold
        ninliers = int(inlier_mask.sum())
        if ninliers > int(best_inlier_mask.sum()):
            best_inlier_mask = inlier_mask.copy()
            best_model = M if model == 'homography' else F
            p_hat = ninliers / float(N)
            if p_hat > 0:
                Pg = p_hat ** m
                # avoid log domain errors
                if Pg >= 1.0:
                    k = 1
                else:
                    k = np.log(1 - confidence) / (np.log(1 - Pg) + 1e-12)
    return best_model, best_inlier_mask, it

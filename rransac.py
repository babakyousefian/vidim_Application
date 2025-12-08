# rransac.py
"""
Robust RANSAC / R-RANSAC implementation with optional SPRT and Td pretests.

Public API:
    rransac_main(pts1, pts2, model='homography', threshold=3.0,
                 confidence=0.99, max_iter=1000, pretest=None, d=1,
                 sprt_params=None, verbose=False)

Returns:
    best_model, best_inliers_mask (bool array), stats (dict)

Notes:
- pts1, pts2: Nx2 numpy arrays of corresponding points (float).
- model: 'homography' or 'fundamental'
- pretest: None, 'SPRT', or 'Td' (case-insensitive)
- sprt_params: dict with keys 'epsilon', 'eta', optionally 'alpha','beta','max_checks'
"""

# existing imports...

# near other pretest imports

try:
    from wald_opt import wald_opt_pretest_candidate
except Exception:
    def wald_opt_pretest_candidate(*args, **kwargs):
        return 'undecided', 0, {}


try:
    from wald import wald_pretest_candidate
except Exception:
    def wald_pretest_candidate(*args, **kwargs):
        return 'undecided', 0, {}


try:
    from bailout import bail_out_pretest_candidate
except Exception:
    def bail_out_pretest_candidate(*args, **kwargs):
        # fallback: no-op
        return 'undecided', 0, {}


try:
    from sprt import sprt_pretest_candidate
except Exception:
    def sprt_pretest_candidate(error_func, candidate, pts1, pts2, threshold, epsilon, eta, alpha=0.01, beta=0.01, max_checks=None):
        return 'undecided', 0, 0.0

# add this (after above)
try:
    from sprt_star import sprt_star_pretest_candidate
except Exception:
    def sprt_star_pretest_candidate(*args, **kwargs):
        # fallback to plain sprt_pretest_candidate if sprt_star not available
        return sprt_pretest_candidate(*args, **kwargs)


import numpy as np
import math
import time
from typing import Tuple, Optional, Dict

# Try to reuse user's provided functions (preferred)
try:
    from model_estimation import estimate_homography, estimate_fundamental, reprojection_error_homography, sampson_error
except Exception:
    # Provide fallback implementations if user module not present.
    def _normalize_points(pts):
        """Normalize points (Nx2) for numeric stability. Returns (norm_pts, T)."""
        pts = np.asarray(pts, dtype=float)
        mean = pts.mean(axis=0)
        std = pts.std(axis=0)
        # use isotropic scale based on average std
        s = np.sqrt(2.0) / (np.mean(std) + 1e-12)
        T = np.array([[s, 0, -s * mean[0]],
                      [0, s, -s * mean[1]],
                      [0, 0, 1]], dtype=float)
        pts_h = np.column_stack([pts, np.ones(len(pts))])
        norm = (T @ pts_h.T).T
        return norm, T

    def estimate_homography(pts1, pts2):
        """
        DLT homography estimation with normalization.
        pts1, pts2: Nx2 arrays. Need at least 4 points.
        Returns 3x3 H (or None on failure).
        """
        pts1 = np.asarray(pts1, dtype=float)
        pts2 = np.asarray(pts2, dtype=float)
        if pts1.shape[0] < 4 or pts2.shape[0] < 4:
            return None
        n1, T1 = _normalize_points(pts1)
        n2, T2 = _normalize_points(pts2)
        N = n1.shape[0]
        A = np.zeros((2 * N, 9))
        for i in range(N):
            x, y = n1[i, 0], n1[i, 1]
            u, v = n2[i, 0], n2[i, 1]
            A[2 * i] = [-x, -y, -1, 0, 0, 0, u * x, u * y, u]
            A[2 * i + 1] = [0, 0, 0, -x, -y, -1, v * x, v * y, v]
        try:
            U, S, Vt = np.linalg.svd(A)
            h = Vt[-1, :]
            Hn = h.reshape(3, 3)
            # denormalize
            H = np.linalg.inv(T2) @ Hn @ T1
            H /= (H[2, 2] + 1e-18)
            return H
        except Exception:
            return None

    def estimate_fundamental(pts1, pts2):
        """
        Normalized 8-point algorithm (or least-squares when >8).
        Enforces rank-2 constraint.
        """
        pts1 = np.asarray(pts1, dtype=float)
        pts2 = np.asarray(pts2, dtype=float)
        if pts1.shape[0] < 8:
            return None
        n1, T1 = _normalize_points(pts1)
        n2, T2 = _normalize_points(pts2)
        N = n1.shape[0]
        A = np.zeros((N, 9))
        for i in range(N):
            x1, y1 = n1[i, 0], n1[i, 1]
            x2, y2 = n2[i, 0], n2[i, 1]
            A[i] = [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1]
        try:
            U, S, Vt = np.linalg.svd(A)
            f = Vt[-1, :]
            Fnorm = f.reshape(3, 3)
            # enforce rank-2
            Uf, Sf, Vtf = np.linalg.svd(Fnorm)
            Sf[-1] = 0.0
            Fnorm = Uf @ np.diag(Sf) @ Vtf
            # denormalize
            F = T2.T @ Fnorm @ T1
            # scale
            F /= (np.linalg.norm(F) + 1e-18)
            return F
        except Exception:
            return None

    def reprojection_error_homography(H, pts1, pts2):
        """
        Compute symmetric reprojection error of homography H for point pairs.
        Returns per-point euclidean error.
        """
        if H is None:
            return np.full(len(pts1), np.inf)
        pts1 = np.asarray(pts1, dtype=float)
        pts2 = np.asarray(pts2, dtype=float)
        N = len(pts1)
        p1h = np.column_stack([pts1, np.ones(N)])
        p2h = np.column_stack([pts2, np.ones(N)])
        Hp1 = (H @ p1h.T).T
        Hp1 = Hp1[:, :2] / (Hp1[:, 2:3] + 1e-12)
        Hinv = None
        try:
            Hinv = np.linalg.inv(H)
        except Exception:
            Hinv = None
        # forward and backward symmetric error
        err_fwd = np.linalg.norm(Hp1 - pts2, axis=1)
        if Hinv is not None:
            Ht2 = (Hinv @ p2h.T).T
            Ht2 = Ht2[:, :2] / (Ht2[:, 2:3] + 1e-12)
            err_bwd = np.linalg.norm(Ht2 - pts1, axis=1)
            return 0.5 * (err_fwd + err_bwd)
        else:
            return err_fwd

    def sampson_error(F, pts1, pts2):
        """
        Sampson distance for fundamental matrix F.
        pts are Nx2.
        """
        if F is None:
            return np.full(len(pts1), np.inf)
        pts1 = np.asarray(pts1, dtype=float)
        pts2 = np.asarray(pts2, dtype=float)
        N = len(pts1)
        p1h = np.column_stack([pts1, np.ones(N)])
        p2h = np.column_stack([pts2, np.ones(N)])
        # l2 = F * p1
        Fx1 = (F @ p1h.T).T  # Nx3
        Ftx2 = (F.T @ p2h.T).T
        denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
        numer = np.sum(p2h * (F @ p1h.T).T, axis=1) ** 2
        return numer / (denom + 1e-12)

# Try to import SPRT pretest helper (if user added sprt.py)
try:
    from sprt import sprt_pretest_candidate
except Exception:
    # fallback: provide a simple placeholder that always returns 'undecided'
    def sprt_pretest_candidate(error_func, candidate, pts1, pts2, threshold, epsilon, eta, alpha=0.01, beta=0.01, max_checks=None):
        # conservative behavior: undecided -> let full verification run
        return 'undecided', 0, 0.0

# Td,d pretest (simple practical variant)
def Td_d_pretest_simple(error_func, candidate, pts1, pts2, threshold, d=1):
    """
    A practical Td,d pretest: sample up to d points (or fewer if not enough),
    check if they are inliers. If all sampled points are inliers -> pass;
    otherwise reject.
    This is a light-weight heuristic/pretest; it's permissive for small d.
    """
    N = len(pts1)
    if N == 0:
        return False
    d = max(1, int(d))
    # if N < d then test all
    idxs = np.random.choice(N, min(d, N), replace=False)
    errs = error_func(candidate, pts1[idxs], pts2[idxs])
    return np.all(errs < threshold)

# R-RANSAC main
def rransac_main(pts1: np.ndarray,
                 pts2: np.ndarray,
                 model: str = 'homography',
                 threshold: float = 3.0,
                 confidence: float = 0.99,
                 max_iter: int = 1000,
                 pretest: Optional[str] = None,
                 d: int = 1,
                 sprt_params: Optional[Dict] = None,
                 verbose: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict]:
    """
    Robust RANSAC / R-RANSAC driver.

    Returns:
        best_model: model parameters (H 3x3 for homography or F 3x3 for fundamental), or None
        best_inliers_mask: boolean array of length N indicating inliers (or None)
        stats: dictionary with diagnostic counters
    """
    t0 = time.time()
    pts1 = np.asarray(pts1, dtype=float)
    pts2 = np.asarray(pts2, dtype=float)
    if pts1.shape != pts2.shape:
        raise ValueError("pts1 and pts2 must have same shape")

    N = len(pts1)
    stats = {
        'N': N,
        'model': model,
        'threshold': threshold,
        'confidence': confidence,
        'max_iter': max_iter,
        'pretest': pretest,
        'd': d,
        'sprt_params': sprt_params,
        # counters
        'candidates_tested': 0,
        'accepted_pretests': 0,
        'rejected_pretests': 0,
        'pretest_checks': 0,
        'full_checks': 0,
        'total_checks': 0,
        'time': None,
    }

    if N < 4 and model == 'homography':
        return None, None, stats
    if N < 8 and model == 'fundamental':
        # allow attempt, but likely to fail
        pass

    # determine minimal sample size m
    if model.lower() == 'homography':
        m = 4
        model_estimator = estimate_homography
        def error_func(candidate, p1, p2):
            return reprojection_error_homography(candidate, p1, p2)
    elif model.lower() == 'fundamental':
        m = 8
        model_estimator = estimate_fundamental
        def error_func(candidate, p1, p2):
            return sampson_error(candidate, p1, p2)
    else:
        raise ValueError("Unsupported model: " + str(model))

    best_model = None
    best_inliers_mask = np.zeros(N, dtype=bool)
    best_inlier_count = 0

    iter_i = 0
    # dynamic iteration bound (RANSAC formula)
    # initial conservative k
    k = max_iter

    rng = np.random.default_rng()

    while iter_i < k and iter_i < max_iter:
        iter_i += 1
        stats['candidates_tested'] += 1

        # 1) draw minimal sample
        if N < m:
            break
        try:
            sample_idxs = rng.choice(N, m, replace=False)
        except Exception:
            sample_idxs = np.random.choice(N, m, replace=False)

        s1 = pts1[sample_idxs]
        s2 = pts2[sample_idxs]

        # 2) estimate candidate model
        candidate = None
        try:
            candidate = model_estimator(s1, s2)
        except Exception:
            candidate = model_estimator(s1, s2)
        if candidate is None:
            # degenerate sample
            continue

        # 3) pretest (optional)
        passed_pretest = True
        if pretest is not None:
            p = str(pretest).lower()
            if p == 'td' or p == 'td,d' or p == 'td_d' or p == 'td,d':
                # run Td,d pretest (simple variant)
                stats['pretest_checks'] += min(d, N)
                passed_pretest = Td_d_pretest_simple(error_func, candidate, pts1, pts2, threshold, d=d)
                if passed_pretest:
                    stats['accepted_pretests'] += 1
                else:
                    stats['rejected_pretests'] += 1
                # continue to next candidate if rejected
                if not passed_pretest:
                    stats['total_checks'] += min(d, N)
                    continue
                else:
                    stats['total_checks'] += min(d, N)
            elif p == 'sprt':
                # require sprt_params
                if not sprt_params:
                    # if not provided, treat as undecided (let full verification run)
                    pass
                else:
                    # sprt_pretest_candidate returns (decision, checks, logL)
                    dec, checks, logL = sprt_pretest_candidate(error_func, candidate, pts1, pts2, threshold,
                                                               sprt_params.get('epsilon', 0.02),
                                                               sprt_params.get('eta', 0.5),
                                                               sprt_params.get('alpha', 0.01),
                                                               sprt_params.get('beta', 0.01),
                                                               sprt_params.get('max_checks', None))
                    stats['pretest_checks'] += checks
                    stats['total_checks'] += checks
                    if dec == 'reject':
                        stats['rejected_pretests'] += 1
                        passed_pretest = False
                        continue
                    elif dec == 'accept':
                        stats['accepted_pretests'] += 1
                        passed_pretest = True
                    else:
                        # undecided: conservatively allow full verification
                        passed_pretest = True

            elif p in ('sprt*', 'sprt_star'):
                # use the SPRT* variant which orders checks by candidate error
                if not sprt_params:
                    pass
                else:
                    dec, checks, logL = sprt_star_pretest_candidate(
                        error_func, candidate, pts1, pts2, threshold,
                        sprt_params.get('epsilon', 0.02),
                        sprt_params.get('eta', 0.5),
                        sprt_params.get('alpha', 0.01),
                        sprt_params.get('beta', 0.01),
                        sprt_params.get('max_checks', None),
                        order_by_candidate_error=True
                    )
                    stats['pretest_checks'] += checks
                    stats['total_checks'] += checks
                    if dec == 'reject':
                        stats['rejected_pretests'] += 1
                        passed_pretest = False
                        continue
                    elif dec == 'accept':
                        stats['accepted_pretests'] += 1
                        passed_pretest = True
                    else:
                        passed_pretest = True

            elif p in ('bail-out', 'bailout', 'bail_out'):
                # Bail-out pretest usage
                # choose parameters sensibly or fetch from sprt_params dict you already pass
                bo_params = sprt_params or {}
                mode = bo_params.get('mode', 'hoeffding')  # 'hoeffding' or 'deterministic'
                required_ratio = bo_params.get('required_inlier_ratio', 0.5)
                delta = bo_params.get('delta', 1e-3)
                max_checks = bo_params.get('max_checks', None)
                required_min_inliers = bo_params.get('required_min_inliers', None)

                dec, checks, info = bail_out_pretest_candidate(
                    error_func, candidate, pts1, pts2, threshold,
                    mode=mode,
                    required_inlier_ratio=required_ratio,
                    required_min_inliers=required_min_inliers,
                    delta=delta,
                    max_checks=max_checks,
                    order_by_candidate_error=True
                )
                # update stats if you have stats counters:
                if 'stats' in locals():
                    stats['pretest_checks'] = stats.get('pretest_checks', 0) + checks
                    stats['total_checks'] = stats.get('total_checks', 0) + checks

                if dec == 'reject':
                    stats['rejected_pretests'] = stats.get('rejected_pretests', 0) + 1
                    passed_pretest = False
                    continue  # skip this candidate
                elif dec == 'accept':
                    stats['accepted_pretests'] = stats.get('accepted_pretests', 0) + 1
                    passed_pretest = True
                else:
                    # undecided -> continue with RANSAC's normal verification (don't automatically reject)
                    passed_pretest = True

            elif p in ('wald',):
                # پارامترها را از sprt_params یا پارامتر پیش‌فرض بگیرید
                wp = sprt_params or {}
                p0 = wp.get('p0', 0.1)
                p1 = wp.get('p1', 0.6)
                alpha = wp.get('alpha', 0.05)
                beta = wp.get('beta', 0.05)
                max_checks = wp.get('max_checks', None)

                dec, checks, info = wald_pretest_candidate(error_func, candidate, pts1, pts2, threshold,
                                                           p0=p0, p1=p1, alpha=alpha, beta=beta,
                                                           max_checks=max_checks, order_by_candidate_error=True)
                # update stats (مثلاً)
                stats['pretest_checks'] = stats.get('pretest_checks', 0) + checks
                if dec == 'reject':
                    stats['rejected_pretests'] = stats.get('rejected_pretests', 0) + 1
                    passed_pretest = False
                    continue
                elif dec == 'accept':
                    stats['accepted_pretests'] = stats.get('accepted_pretests', 0) + 1
                    passed_pretest = True
                else:
                    passed_pretest = True  # undecided => fall back to full verification

            elif p in ('wald_opt', 'wald-opt', 'waldopt', 'wald_opt'.lower()):
                wp = sprt_params or {}
                p0 = wp.get('p0', 0.1)
                p1 = wp.get('p1', 0.6)
                alpha = wp.get('alpha', 0.05)
                beta = wp.get('beta', 0.05)
                max_checks = wp.get('max_checks', min(500, len(pts1)))
                min_accept_checks = wp.get('min_accept_checks', 3)
                # ordering options
                order_mode = wp.get('order_mode', 'gain')
                dec, checks, info = wald_opt_pretest_candidate(error_func,
                                                               candidate,
                                                               pts1, pts2,
                                                               threshold,
                                                               p0=p0, p1=p1, alpha=alpha, beta=beta,
                                                               max_checks=max_checks,
                                                               min_accept_checks=min_accept_checks,
                                                               order_mode=order_mode, order_by_candidate_error=True)
                stats['pretest_checks'] = stats.get('pretest_checks', 0) + checks
                if dec == 'reject':
                    passed_pretest = False
                    stats['rejected_pretests'] = stats.get('rejected_pretests', 0) + 1
                    continue
                elif dec == 'accept':
                    passed_pretest = True
                else:
                    # undecided -> fall through to normal verification
                    passed_pretest = True

            else:
                # unknown pretest key: ignore
                pass

        # 4) Full verification (compute errors for all points; vectorized)
        stats['full_checks'] += 1
        errs = error_func(candidate, pts1, pts2)
        stats['total_checks'] += N
        # inlier mask
        in_mask = errs < threshold
        nin = int(in_mask.sum())

        # 5) bail-out optimization:
        # if even assuming all currently-out points were inliers we cannot beat best, skip update
        # i.e., maximum possible inliers for this candidate = nin + (N - checked) but we checked all -> here checked=N
        # However we can use a conservative check: if nin <= best_inlier_count -> skip expensive recompute (already cheap)
        if nin <= best_inlier_count:
            # nothing to do
            continue

        # 6) Update best
        best_inlier_count = nin
        best_inliers_mask = in_mask.copy()
        best_model = candidate

        # 7) recompute required iterations (RANSAC formula)
        # probability that a random m-sample is all-inlier: p = (p_hat)^m
        p_hat = max(1e-12, best_inlier_count / float(N))
        prob_all_inliers = p_hat ** m
        if prob_all_inliers > 0:
            # derive required k given confidence: 1 - (1 - p^m)^k >= confidence -> k >= log(1-confidence)/log(1-p^m)
            denom = math.log(max(1e-12, 1.0 - prob_all_inliers))
            if denom == 0:
                k = 1
            else:
                k_new = math.log(1.0 - confidence + 1e-18) / denom
                # numeric safety
                if k_new < 1:
                    k_new = 1
                # set k to min of current and new (we aim to reduce iterations if possible)
                k = int(math.ceil(min(k, k_new)))
                # ensure k is at least iter_i
                k = max(k, iter_i)
        # continue loop until iter_i >= k or iter_i >= max_iter

    stats['time'] = time.time() - t0
    stats['candidates_tested'] = iter_i
    stats['best_inlier_count'] = int(best_inlier_count)
    stats['best_inlier_ratio'] = (best_inlier_count / float(N) if N > 0 else 0.0)
    stats['best_inliers_mask_sum'] = int(best_inliers_mask.sum()) if best_inliers_mask is not None else 0
    stats['total_checks'] = stats.get('total_checks', 0)
    stats['full_checks'] = stats.get('full_checks', 0)
    stats['pretest_checks'] = stats.get('pretest_checks', 0)
    stats['accepted_pretests'] = stats.get('accepted_pretests', 0)
    stats['rejected_pretests'] = stats.get('rejected_pretests', 0)

    # finalize: best_inliers_mask as numpy boolean array or None
    if best_model is None:
        return None, None, stats
    else:
        return best_model, best_inliers_mask.astype(bool), stats

# small self-test when run directly
if __name__ == "__main__":
    # generate synthetic homography example to sanity-check
    np.random.seed(0)
    # create random points
    N = 200
    pts = np.random.uniform(0, 500, size=(N, 2))
    # true homography: scale + translate + small rotation
    angle = 0.05
    s = 1.02
    tx, ty = 10.0, -5.0
    R = np.array([[s * math.cos(angle), -s * math.sin(angle), tx],
                  [s * math.sin(angle), s * math.cos(angle), ty],
                  [0, 0, 1]])
    pts_h = np.column_stack([pts, np.ones(N)])
    pts2h = (R @ pts_h.T).T
    pts2 = pts2h[:, :2] / pts2h[:, 2:3]
    # add noise to some points and outliers
    pts2_noisy = pts2.copy()
    # add gaussian noise to all
    pts2_noisy += np.random.normal(scale=0.5, size=pts2_noisy.shape)
    # introduce outliers
    n_out = int(0.3 * N)
    out_idx = np.random.choice(N, n_out, replace=False)
    pts2_noisy[out_idx] = np.random.uniform(0, 500, size=(n_out, 2))

    H, mask, st = rransac_main(pts, pts2_noisy, model='homography', threshold=3.0, confidence=0.99, max_iter=2000, pretest=None)
    print("Test run stats:", st)
    if H is not None:
        print("Estimated H:\n", H)

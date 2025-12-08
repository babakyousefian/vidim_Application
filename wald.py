# wald.py
"""
Wald sequential test pretest for RANSAC-like candidate verification.

Public API:
    wald_pretest_candidate(error_func, candidate, pts1, pts2, threshold,
                           p0=0.1, p1=0.6, alpha=0.05, beta=0.05,
                           max_checks=None, order_by_candidate_error=True)

Returns:
    (decision, checks, info)
    decision in {'accept', 'reject', 'undecided'}
    checks: number of points checked
    info: dict with keys 'llr', 'inliers', 'checks', 'p0','p1','A','B'
Notes:
- error_func(candidate, p1_subset, p2_subset) should return array-like errors for given pairs
- Xi = 1 if error < threshold else 0
- LLR incremental update:
    LLR += Xi * log(p1/p0) + (1-Xi) * log((1-p1)/(1-p0))
- Boundaries:
    A = log((1-beta)/alpha)    -> accept H1 when LLR >= A
    B = log(beta/(1-alpha))    -> accept H0 (reject H1) when LLR <= B
"""
from typing import Tuple, Dict
import numpy as np
import math

def _safe_log(x):
    # avoid log(0)
    eps = 1e-300
    return math.log(max(x, eps))

def wald_pretest_candidate(error_func,
                           candidate,
                           pts1,
                           pts2,
                           threshold,
                           p0: float = 0.1,
                           p1: float = 0.6,
                           alpha: float = 0.05,
                           beta: float = 0.05,
                           max_checks: int = None,
                           order_by_candidate_error: bool = True) -> Tuple[str, int, Dict]:
    """
    Perform sequential Wald test on correspondences for candidate model.

    Parameters:
      error_func: callable(candidate, pts1_subset, pts2_subset) -> array(errors)
      candidate: model (H or F)
      pts1, pts2: arrays (N x 2)
      threshold: numeric threshold for inlier classification
      p0: inlier probability under H0 (bad model)
      p1: inlier probability under H1 (good model)
      alpha: desired false positive rate (accept H1 when H0 true)
      beta: desired false negative rate (reject H1 when H1 true)
      max_checks: maximum number of sequential checks (default N)
      order_by_candidate_error: whether to evaluate low-error points first
    Returns:
      decision ('accept'|'reject'|'undecided'), checks, info
    """
    pts1 = np.asarray(pts1)
    pts2 = np.asarray(pts2)
    N = len(pts1)
    if N == 0:
        return 'undecided', 0, {'inliers': 0, 'checks': 0}

    if max_checks is None:
        max_checks = N
    else:
        max_checks = min(max_checks, N)

    # precompute log-likelihood ratios per binary outcome
    # For Xi = 1 (inlier):
    log_inlier = _safe_log(p1) - _safe_log(p0)
    # For Xi = 0 (outlier):
    log_outlier = _safe_log(1.0 - p1) - _safe_log(1.0 - p0)

    # decision thresholds
    A = _safe_log((1.0 - beta) / max(alpha, 1e-300))   # accept H1 when LLR >= A
    B = _safe_log(max(beta, 1e-300) / max(1.0 - alpha, 1e-300))  # accept H0 when LLR <= B

    # ordering by predicted error (more informative first)
    order = np.arange(N)
    if order_by_candidate_error:
        try:
            errs = error_func(candidate, pts1, pts2)
            if len(errs) == N:
                order = np.argsort(errs)  # smallest errors first
        except Exception:
            order = np.arange(N)

    llr = 0.0
    checks = 0
    inliers = 0

    for i in range(min(max_checks, N)):
        idx = order[i]
        # get error for this point
        try:
            e = error_func(candidate, pts1[[idx]], pts2[[idx]])
            err_val = float(e[0])
        except Exception:
            # fallback: compute all errs once
            try:
                all_errs = error_func(candidate, pts1, pts2)
                err_val = float(all_errs[idx])
            except Exception:
                return 'undecided', checks, {'inliers': inliers, 'checks': checks, 'llr': llr}

        checks += 1
        is_inlier = (err_val < threshold)
        if is_inlier:
            inliers += 1
            llr += log_inlier
        else:
            llr += log_outlier

        # immediate decision checks
        if llr >= A:
            return 'accept', checks, {'inliers': inliers, 'checks': checks, 'llr': llr, 'A': A, 'B': B, 'p0': p0, 'p1': p1}
        if llr <= B:
            return 'reject', checks, {'inliers': inliers, 'checks': checks, 'llr': llr, 'A': A, 'B': B, 'p0': p0, 'p1': p1}

    # exhausted checks without crossing boundaries
    # conservative final decision: if empirical ratio strongly favors H1 we may accept, else undecided
    if checks > 0:
        empirical = float(inliers) / float(checks)
        # if empirical >= p1 (or close) -> accept (conservative)
        if empirical >= p1:
            return 'accept', checks, {'inliers': inliers, 'checks': checks, 'llr': llr, 'empirical': empirical}
        # if empirical <= p0 -> reject
        if empirical <= p0:
            return 'reject', checks, {'inliers': inliers, 'checks': checks, 'llr': llr, 'empirical': empirical}
    return 'undecided', checks, {'inliers': inliers, 'checks': checks, 'llr': llr}

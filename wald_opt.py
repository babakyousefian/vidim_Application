# wald_opt.py
"""
Wald-Opt (optimized Wald / SPRT-style pretest) for RANSAC candidate verification.

Design goals / features:
- Same public API shape as wald.wald_pretest_candidate for easy swap-in.
- Uses adaptive ordering: evaluate points in order of *expected information gain*.
  We approximate information gain by transforming errors -> scores and sorting by
  how strongly they suggest inlier vs outlier.
- Uses truncated checks: optionally require a minimum number of checks before making
  an 'accept' decision to avoid false accepts from a small lucky sample.
- Uses safe logs to avoid numerical issues.
- Returns same tuple: (decision, checks, info) where decision in {'accept','reject','undecided'}.
- Parameters:
    p0,p1,alpha,beta: same meaning as in Wald
    max_checks: upper cap on sequential checks
    min_accept_checks: minimum number of checks required before accepting H1 (helps stability)
    order_mode: 'gain' (default) or 'error' (fallback to simple ascending error)
    score_transform: 'logodds' (default) map p->score for ordering; or callable

Public function:
    wald_opt_pretest_candidate(error_func, candidate, pts1, pts2, threshold, **kwargs)
"""

from typing import Tuple, Dict, Callable
import numpy as np
import math

def _safe_log(x: float) -> float:
    eps = 1e-300
    return math.log(max(x, eps))

def _score_from_error(err: float, threshold: float) -> float:
    """
    Map error -> score in (-inf..+inf): negative for likely outlier, positive for likely inlier.
    We use a simple crushed linear mapping around threshold.
    """
    # small margin scaling
    if err is None:
        return 0.0
    # distance relative to threshold
    d = (threshold - err) / max(1e-6, threshold)
    # squashed by atanh-like curve to emphasize confident points
    # keep numerically stable: use sign-preserving log1p
    return math.copysign(math.log1p(abs(d) * 10.0), d)

def _default_score_transform(errs, threshold):
    # produce scores for all errors
    return np.array([_score_from_error(float(e), threshold) for e in errs])

def wald_opt_pretest_candidate(error_func: Callable,
                               candidate,
                               pts1,
                               pts2,
                               threshold: float,
                               p0: float = 0.1,
                               p1: float = 0.6,
                               alpha: float = 0.05,
                               beta: float = 0.05,
                               max_checks: int = None,
                               min_accept_checks: int = 3,
                               order_mode: str = 'gain',
                               score_transform = None,
                               order_by_candidate_error: bool = True
                               ) -> Tuple[str, int, Dict]:
    """
    Wald-Opt pretest.

    Parameters:
      error_func: callable(candidate, pts1_subset, pts2_subset) -> array(errors)
      candidate: model (H or F)
      pts1, pts2: arrays (N x 2)
      threshold: numeric threshold for inlier classification
      p0, p1, alpha, beta: Wald parameters
      max_checks: maximum sequential checks (default N)
      min_accept_checks: don't ACCEPT H1 before this many checks (helps avoid small-sample false accepts)
      order_mode: 'gain' (default) attempts to order by information gain (via score_transform),
                  'error' sorts by smallest error first (fallback)
      score_transform: callable(errs, threshold) -> scores array, or None to use default
      order_by_candidate_error: if False, keep original order (rare)
    Returns:
      decision ('accept'|'reject'|'undecided'), checks, info dict
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

    # logs for outcomes
    log_inlier = _safe_log(p1) - _safe_log(p0)
    log_outlier = _safe_log(1.0 - p1) - _safe_log(1.0 - p0)

    # decision thresholds
    A = _safe_log((1.0 - beta) / max(alpha, 1e-300))   # accept H1 when LLR >= A
    B = _safe_log(max(beta, 1e-300) / max(1.0 - alpha, 1e-300))  # accept H0 when LLR <= B

    # compute ordering indices
    order = np.arange(N)
    if order_by_candidate_error:
        # try to obtain per-point errors
        try:
            errs = error_func(candidate, pts1, pts2)
            errs = np.asarray(errs, dtype=float)
            if score_transform is None:
                score_transform = _default_score_transform
            # compute scores
            try:
                scores = score_transform(errs, threshold)
                # Information-directed ordering: points with largest |score| first,
                # and prefer positive scores (inlier-like) first to allow quick acceptance.
                order = np.argsort(-np.abs(scores) * np.sign(scores))  # prefer strong positive, then strong negative
                # fallback if order invalid:
                if len(order) != N:
                    order = np.argsort(errs)  # small error first
            except Exception:
                # fallback simple error ordering
                order = np.argsort(errs)
        except Exception:
            order = np.arange(N)

    llr = 0.0
    checks = 0
    inliers = 0

    for i in range(min(max_checks, N)):
        idx = order[i]
        # fetch error for single point
        try:
            e = error_func(candidate, pts1[[idx]], pts2[[idx]])
            err_val = float(e[0])
        except Exception:
            # last-resort: compute all errs then index
            try:
                all_errs = error_func(candidate, pts1, pts2)
                err_val = float(all_errs[idx])
            except Exception:
                # cannot compute; abort undecided
                return 'undecided', checks, {'inliers': inliers, 'checks': checks, 'llr': llr}

        checks += 1
        is_inlier = (err_val < threshold)
        if is_inlier:
            inliers += 1
            llr += log_inlier
        else:
            llr += log_outlier

        # immediate decision checks
        if llr >= A and checks >= min_accept_checks:
            return 'accept', checks, {'inliers': inliers, 'checks': checks, 'llr': llr, 'A': A, 'B': B, 'p0': p0, 'p1': p1}
        if llr <= B:
            return 'reject', checks, {'inliers': inliers, 'checks': checks, 'llr': llr, 'A': A, 'B': B, 'p0': p0, 'p1': p1}

    # exhausted without reaching either threshold
    # Decide conservatively: use empirical fraction with margin (and require min_accept_checks)
    if checks > 0:
        empirical = float(inliers) / float(checks)
        # if empirical strongly favors H1 and we had at least min_accept_checks -> accept
        if empirical >= max(p1, 0.75) and checks >= min_accept_checks:
            return 'accept', checks, {'inliers': inliers, 'checks': checks, 'llr': llr, 'empirical': empirical}
        # if empirical strongly below p0 -> reject
        if empirical <= min(p0, 0.25):
            return 'reject', checks, {'inliers': inliers, 'checks': checks, 'llr': llr, 'empirical': empirical}
    return 'undecided', checks, {'inliers': inliers, 'checks': checks, 'llr': llr}

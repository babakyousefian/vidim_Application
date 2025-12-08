# bail_out.py
"""
Bail-out pretest helper for R-RANSAC.

Two modes implemented:
  1) deterministic_min (logical early-reject): if current_inliers + remaining_possible < required_min_inliers -> reject.
  2) statistical_hoeffding: uses Hoeffding inequality to compute an upper bound on the final inlier ratio
     and rejects early if that upper bound is below required_ratio with confidence (1 - delta).

Public API:
  bail_out_pretest_candidate(error_func, candidate, pts1, pts2, threshold,
                              mode='hoeffding',
                              required_inlier_ratio=0.5,
                              required_min_inliers=None,
                              delta=1e-3,
                              max_checks=None,
                              order_by_candidate_error=True)

Arguments:
- error_func(candidate, p1_subset, p2_subset) -> array(errors)  (vectorized)
- candidate: model to evaluate (H or F)
- pts1, pts2: Nx2 arrays of correspondences
- threshold: inlier threshold (error < threshold => inlier)
- mode: 'deterministic' or 'hoeffding' (default 'hoeffding')
- required_inlier_ratio: fraction of points considered necessary for acceptance (tau). Used in hoeffding mode.
- required_min_inliers: absolute minimal inlier count for acceptance (if provided, used in deterministic check)
- delta: acceptable probability of wrong rejection in hoeffding mode (confidence parameter)
- max_checks: maximum sequential checks; if None -> all points potentially checked (but algorithm will try to decide earlier)
- order_by_candidate_error: if True, check points with smallest predicted error first (more informative first).
Return:
- (decision, checks, info)
  decision in {'accept', 'reject', 'undecided'}
  checks = number of points actually checked
  info = dict with keys: 'inliers', 'checks', 'log' (optional)
Notes:
- This helper is deliberately conservative: 'accept' is returned only if checks exhausted and ratio >= required_ratio,
  otherwise 'undecided' is returned (so final rransac logic still validates candidate).
- Use required_inlier_ratio/tau sensibly for dataset size & expected inlier fraction.
"""

from typing import Tuple, Dict
import numpy as np
import math

# numerical safe log + clamp helpers
def _safe_frac(x):
    return min(max(x, 0.0), 1.0)

def _hoeffding_upper_bound(mean_observed: float, n_checked: int, n_total: int, delta: float) -> float:
    """
    Hoeffding upper confidence bound for true mean p, given observed sample mean.
    Using Hoeffding inequality: P( p - mean_obs >= t ) <= exp(-2 * n_checked * t^2)
    Solve for t: t = sqrt( -log(delta) / (2 * n_checked) )
    Upper bound on p = mean_obs + t
    If n_checked == 0 -> return 1.0 (no info)
    """
    if n_checked <= 0:
        return 1.0
    t = math.sqrt(max(-math.log(max(delta, 1e-300)) / (2.0 * n_checked), 0.0))
    return min(mean_observed + t, 1.0)


def bail_out_pretest_candidate(error_func,
                               candidate,
                               pts1,
                               pts2,
                               threshold,
                               mode: str = 'hoeffding',
                               required_inlier_ratio: float = 0.5,
                               required_min_inliers: int = None,
                               delta: float = 1e-3,
                               max_checks: int = None,
                               order_by_candidate_error: bool = True) -> Tuple[str, int, Dict]:
    """
    Run Bail-out pretest and return (decision, checks, info).
    Decision: 'accept' (strong evidence), 'reject' (strong evidence to reject), 'undecided' otherwise.
    """
    pts1 = np.asarray(pts1)
    pts2 = np.asarray(pts2)
    N = len(pts1)
    if N == 0:
        return 'undecided', 0, {'inliers':0, 'checks':0}

    # defaults
    if max_checks is None:
        max_checks = N
    else:
        max_checks = min(max_checks, N)

    # ordering (Bail-out heuristic: evaluate low-error points first -> most informative)
    order = np.arange(N)
    if order_by_candidate_error:
        try:
            errs = error_func(candidate, pts1, pts2)
            if len(errs) == N:
                order = np.argsort(errs)  # increasing error -> likely inliers first
        except Exception:
            order = np.arange(N)

    # counters
    checks = 0
    inliers = 0

    # required absolute inliers (if provided)
    req_min = required_min_inliers
    if req_min is None:
        # derive from required_inlier_ratio if given
        req_min = int(math.ceil(required_inlier_ratio * N))

    # fast deterministic rejection check function
    def deterministic_reject_possible(checked, inliers_so_far, remaining):
        """
        If even assuming all remaining points are inliers we still cannot reach req_min -> reject.
        checked: points tested so far
        inliers_so_far: count of inliers among them
        remaining: N - checked
        """
        best_possible_final = inliers_so_far + remaining
        return best_possible_final < req_min

    # main loop
    for i_idx in range(min(max_checks, N)):
        idx = order[i_idx]
        # compute error for this point (try to reuse vectorized function, but call on singleton)
        try:
            e = error_func(candidate, pts1[[idx]], pts2[[idx]])
            val = float(e[0])
        except Exception:
            # fallback: compute all errors once (costly) and index
            try:
                all_errs = error_func(candidate, pts1, pts2)
                val = float(all_errs[idx])
            except Exception:
                # if error computation totally fails -> undecided
                return 'undecided', checks, {'inliers': inliers, 'checks': checks}

        checks += 1
        is_inlier = (val < threshold)
        if is_inlier:
            inliers += 1

        remaining = N - checks

        # Mode-specific early decisions:
        if mode == 'deterministic':
            # deterministic: reject if cannot possibly reach required inliers
            if deterministic_reject_possible(checks, inliers, remaining):
                return 'reject', checks, {'inliers': inliers, 'checks': checks}
            # accept only if we already have >= req_min and we can accept early (optional)
            # Conservative choice: require full scan to accept (so acceptance only at end)
            continue

        elif mode == 'hoeffding':
            # compute observed inlier ratio so far
            mean_obs = float(inliers) / float(checks)
            # compute Hoeffding upper bound on true ratio
            p_upper = _hoeffding_upper_bound(mean_obs, checks, N, delta)
            # If even the optimistic upper bound is below required ratio -> reject
            if p_upper < required_inlier_ratio:
                return 'reject', checks, {'inliers': inliers, 'checks': checks, 'p_upper': p_upper}
            # Optionally: if lower bound above required ratio -> accept early
            # Hoeffding lower bound:
            # P( mean_obs - p >= t ) <= exp(-2 n t^2) -> lower bound = mean_obs - t
            if checks > 0:
                t = math.sqrt(max(-math.log(max(delta, 1e-300)) / (2.0 * checks), 0.0))
                p_lower = max(mean_obs - t, 0.0)
                if p_lower >= required_inlier_ratio:
                    # strong evidence candidate is good
                    return 'accept', checks, {'inliers': inliers, 'checks': checks, 'p_lower': p_lower}
            # else continue checking

        else:
            # unknown mode -> undecided
            return 'undecided', checks, {'inliers': inliers, 'checks': checks, 'error': 'unknown_mode'}

    # exhausted checks (or max_checks)
    # make final decision conservatively: if empirical ratio meets required -> accept, else undecided (caller may reject)
    final_ratio = float(inliers) / float(checks) if checks > 0 else 0.0
    if final_ratio >= required_inlier_ratio:
        return 'accept', checks, {'inliers': inliers, 'checks': checks, 'final_ratio': final_ratio}
    else:
        # Not strong enough to accept; caller may still run full evaluation / use other criteria.
        return 'undecided', checks, {'inliers': inliers, 'checks': checks, 'final_ratio': final_ratio}

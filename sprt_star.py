# sprt_star.py
"""
SPRT* pretest helper for R-RANSAC.

Public function:
    sprt_star_pretest_candidate(error_func, candidate, pts1, pts2, threshold,
                                epsilon, eta, alpha=0.01, beta=0.01, max_checks=None,
                                order_by_candidate_error=True)

- error_func(candidate, p1_subset, p2_subset) -> array(errors)  (vectorized)
- candidate: model parameters (H or F)
- pts1, pts2: full Nx2 arrays of correspondences
- threshold: inlier threshold used to compare error < threshold
- epsilon: model parameter (expected inlier noise level effect). See discussion below.
- eta: nominal probability of being an inlier under null (outlier) hypothesis (small).
- alpha: acceptable false-accept probability (type I)
- beta: acceptable false-reject probability (type II)
- max_checks: maximum number of sequential checks before giving up (None -> use N)
- order_by_candidate_error: if True (default), inspect points ordered by increasing error
  (most-informative first). This is the *star* heuristic.
Return:
    (decision, checks, logLR)
    decision in {'accept','reject','undecided'}
    checks = number of point checks performed
    logLR = final log-likelihood-ratio value
Notes:
- This implementation uses a Bernoulli observation model: each checked point is either an inlier (I)
  or outlier (O). Under H1 (candidate is good) we assume P(I) = p1 = 1 - epsilon.
  Under H0 (candidate is bad) we assume P(I) = p0 = eta.
  For each observation x ∈ {1 if inlier, 0 if outlier} the LR increment is:
      log( p1/p0 ) if x==1
      log( (1-p1)/(1-p0) ) if x==0
- Decision thresholds for SPRT are:
      A = log((1 - beta) / alpha)   -> accept H1 if logLR >= A
      B = log(beta / (1 - alpha))   -> reject H1 if logLR <= B
  (If undecided after max_checks -> return 'undecided')
"""

from typing import Tuple
import numpy as np
import math

def _safe_log(x):
    # avoid -inf on 0
    return math.log(max(x, 1e-300))

def sprt_star_pretest_candidate(error_func,
                                candidate,
                                pts1,
                                pts2,
                                threshold,
                                epsilon,
                                eta,
                                alpha=0.01,
                                beta=0.01,
                                max_checks=None,
                                order_by_candidate_error: bool = True) -> Tuple[str, int, float]:
    """
    Run SPRT* sequential pretest on candidate. Returns (decision, checks, logLR).
    """
    pts1 = np.asarray(pts1)
    pts2 = np.asarray(pts2)
    N = len(pts1)
    if N == 0:
        return 'undecided', 0, 0.0

    # probabilities under hypotheses:
    # H1 (candidate is good): p1 = P(point is inlier | H1)
    # H0 (candidate is bad):  p0 = P(point is inlier | H0)
    # Use p1 = 1 - epsilon, p0 = eta
    p1 = min(max(1.0 - float(epsilon), 1e-6), 1.0 - 1e-6)
    p0 = min(max(float(eta), 1e-9), 1.0 - 1e-9)

    # precompute per-sample log increments
    # when observation = inlier (x=1): inc = log(p1/p0)
    # when observation = outlier (x=0): inc = log((1-p1)/(1-p0))
    inc_inlier = _safe_log(p1) - _safe_log(p0)
    inc_outlier = _safe_log(1.0 - p1) - _safe_log(1.0 - p0)

    # thresholds (log-LR)
    A = _safe_log((1.0 - beta) / max(alpha, 1e-12))    # accept H1 if logLR >= A
    B = _safe_log(beta / max((1.0 - alpha), 1e-12))   # reject H1 if logLR <= B

    # ordering: SPRT* heuristic: evaluate most-likely-inlier points first
    order = np.arange(N)
    if order_by_candidate_error:
        try:
            errs = error_func(candidate, pts1, pts2)
            # ensure shape N
            if len(errs) == N:
                # sort by increasing error (small error -> likely inlier)
                order = np.argsort(errs)
            else:
                # fallback: use plain order
                order = np.arange(N)
        except Exception:
            order = np.arange(N)

    # max_checks bound
    if max_checks is None:
        max_checks = N
    else:
        max_checks = min(max_checks, N)

    logLR = 0.0
    checks = 0

    # iterate sequentially
    for idx in order[:max_checks]:
        # compute error for this single point (call error_func on singletons to reuse user's vectorized code)
        try:
            e = error_func(candidate, pts1[[idx]], pts2[[idx]])
            # e is array-like with one element
            val = float(e[0])
        except Exception:
            # if error function fails for singletons, try computing all errors then index
            try:
                all_errs = error_func(candidate, pts1, pts2)
                val = float(all_errs[idx])
            except Exception:
                # give up: undecided
                return 'undecided', checks, logLR

        is_inlier = (val < threshold)
        if is_inlier:
            logLR += inc_inlier
        else:
            logLR += inc_outlier
        checks += 1

        # decision check
        if logLR >= A:
            return 'accept', checks, logLR
        if logLR <= B:
            return 'reject', checks, logLR

    # if we ran out of checks without decision -> undecided
    return 'undecided', checks, logLR


# quick self-check (very small)
if __name__ == "__main__":
    import numpy as np
    def dummy_err(model, p1s, p2s):
        # simple distance
        return np.linalg.norm(p1s - p2s, axis=1)
    # construct synthetic
    N = 100
    pts1 = np.random.rand(N,2)*100
    # candidate perfect transform: pts2 = pts1
    pts2 = pts1.copy()
    dec, c, lr = sprt_star_pretest_candidate(dummy_err, None, pts1, pts2, threshold=1.0,
                                            epsilon=0.02, eta=0.01, alpha=0.01, beta=0.01)
    print("Decision (should be accept):", dec, "checks:", c, "logLR:", lr)

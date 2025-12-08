# sprt.py
import numpy as np
import math
from typing import Callable, Tuple

def sprt_decision_stream(error_getter: Callable[[int], float],
                         model,
                         threshold: float,
                         epsilon: float,
                         eta: float,
                         alpha: float = 0.01,
                         beta: float = 0.01,
                         max_checks: int = None) -> Tuple[str, int, float]:
    """
    Sequential Probability Ratio Test (binary decision) for inlier/outlier stream.
    error_getter(i) -> scalar error for i-th correspondence (0..N-1)
    We treat x_i = 1 if error < threshold (observed inlier).
    Hypotheses:
      H_g (good model) : P(x=1) = eta
      H_b (bad  model) : P(x=1) = epsilon
    Returns:
      decision: 'accept'|'reject'|'undecided' (undecided only if max_checks reached)
      checks_done: number of checks performed
      logLambda: final log-likelihood ratio (log P(data|H_g)/P(data|H_b))
    """
    # boundaries
    A = (1 - beta) / (alpha + 1e-18)
    B = (beta + 1e-18) / (1 - alpha)
    logA = math.log(A)
    logB = math.log(B)

    # precompute increments
    p1 = eta
    q1 = epsilon
    # protect
    p1 = max(min(p1, 1 - 1e-12), 1e-12)
    q1 = max(min(q1, 1 - 1e-12), 1e-12)

    # increments for observation x==1 or x==0
    inc1 = math.log(p1 / q1)
    inc0 = math.log((1 - p1) / (1 - q1))

    logLR = 0.0
    checks = 0

    # iterate over stream (caller decides ordering)
    i = 0
    while True:
        if max_checks is not None and checks >= max_checks:
            return 'undecided', checks, logLR
        try:
            x = 1 if (error_getter(i) < threshold) else 0
        except IndexError:
            # no more points
            return ('accept' if logLR >= logA else ('reject' if logLR <= logB else 'undecided')), checks, logLR
        # update
        logLR += inc1 if x == 1 else inc0
        checks += 1
        i += 1
        # decision
        if logLR >= logA:
            return 'accept', checks, logLR
        if logLR <= logB:
            return 'reject', checks, logLR

def sprt_pretest_candidate(error_func: Callable[[object, np.ndarray, np.ndarray], np.ndarray],
                           candidate,
                           pts1: np.ndarray,
                           pts2: np.ndarray,
                           threshold: float,
                           epsilon: float,
                           eta: float,
                           alpha: float = 0.01,
                           beta: float = 0.01,
                           max_checks: int = None) -> Tuple[str, int, float]:
    """
    Runs SPRT sequential test over the correspondences for a candidate model.
    error_func(candidate, pts1, pts2) -> per-point error array (vectorized)
    Returns (decision, checks_done, logL)
    The order of checks is randomized to avoid bias (common practice).
    """
    N = len(pts1)
    if N == 0:
        return 'reject', 0, 0.0

    idxs = np.arange(N)
    np.random.shuffle(idxs)

    # Make error_getter closure that uses the random order
    errs = error_func(candidate, pts1, pts2)

    def getter(i):
        if i >= N:
            raise IndexError('stream exhausted')
        return float(errs[idxs[i]])

    return sprt_decision_stream(getter, candidate, threshold, epsilon, eta, alpha, beta, max_checks)

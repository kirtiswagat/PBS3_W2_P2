"""Conformal risk control for false-negative rate (H5's clinically load-bearing guarantee).

Plain conformal prediction controls the coverage of a SET; for a screening
task, the guarantee that matters is a distribution-free bound on MISSED
findings, i.e. FNR. This follows Angelopoulos et al. (2022) conformal risk
control: FNR is monotone non-decreasing in the decision threshold lambda (a
higher threshold means fewer positive calls, so more misses), so the
finite-sample-corrected empirical-risk estimate is scanned from the most
sensitive threshold upward, and the last one that still clears alpha is kept.
"""

import numpy as np


def conformal_risk_control_fnr(cal_p, cal_y, test_p, alpha=0.1, n_thresholds=200):
    cal_p, cal_y, test_p = np.asarray(cal_p), np.asarray(cal_y), np.asarray(test_p)
    n = len(cal_y)
    thresholds = np.unique(np.quantile(cal_p, np.linspace(0.0, 1.0, n_thresholds)))
    pos = cal_y == 1
    n_pos = int(pos.sum())

    def fnr_at(lam):
        if n_pos == 0:
            return 0.0
        return float(((cal_p < lam) & pos).sum() / n_pos)

    # (n / (n+1)) * R_hat(lambda) + 1/(n+1) <= alpha is the finite-sample
    # correction; FNR is non-decreasing in lambda, so the first threshold (in
    # ascending order) that fails means every larger one fails too.
    best_lambda, achieved = thresholds[0], False
    for lam in thresholds:
        corrected = (n / (n + 1)) * fnr_at(lam) + 1.0 / (n + 1)
        if corrected <= alpha:
            best_lambda, achieved = lam, True
        else:
            break

    if not achieved:
        print(
            f"conformal_risk_control_fnr: WARNING - no threshold satisfies "
            f"FNR<=alpha={alpha} on this calibration set (n_pos={n_pos}); "
            f"falling back to the most sensitive threshold"
        )

    empirical_fnr = fnr_at(best_lambda)
    test_pred_pos = test_p >= best_lambda
    return {
        "threshold": float(best_lambda),
        "empirical_fnr": empirical_fnr,
        "coverage": float(1.0 - empirical_fnr),
        # A single threshold yields a point decision, not a variable-size set;
        # kept as 1.0 for interface symmetry with split/mondrian conformal.
        "avg_set_size": 1.0,
        "test_flagged_rate": float(test_pred_pos.mean()),
    }

"""Conformal coverage under covariate shift (H5): CheXpert -> NIH.

Split conformal's 1-alpha coverage guarantee assumes calibration and test
rows are exchangeable. Train-B and Train-C satisfy that (same source, same
sampling). Train-B and NIH almost certainly do not: different scanners,
different population, different labelling pipeline. Coverage on NIH is
therefore PREDICTED to fall below 1-alpha - that violation is the result, not
a bug, and is reported alongside the mitigation.

WEIGHTED CONFORMAL AS MITIGATION
    Tibshirani et al. (2019): if the covariate-shift likelihood ratio
    w(x) = p_external(x) / p_internal(x) is known (or estimated), reweighting
    the calibration quantile by w restores the guarantee under the shifted
    distribution. The ratio is estimated here with a simple domain classifier
    (logistic regression discriminating Train-B rows from NIH rows on the
    standardised embedding) rather than assumed known.
"""

import numpy as np

from conformal import binary_nonconformity, split_conformal


def coverage_under_shift(cal_b, test_internal, test_external, alpha=0.1, x_cal=None, x_external=None):
    """cal_b, test_internal, test_external are each (p, y) tuples of arrays.

    Calibrates a threshold on cal_b (Train-B), then checks empirical coverage
    on test_internal (Train-C) and test_external (NIH) against the SAME
    threshold. If x_cal/x_external (standardised emb_backbone) are supplied,
    also runs the weighted-conformal mitigation and reports recovered coverage.
    """
    cal_p, cal_y = cal_b
    cal_y = np.asarray(cal_y).astype(int)
    cal_scores = binary_nonconformity(cal_p)[np.arange(len(cal_y)), cal_y]

    test_p_int, test_y_int = test_internal
    test_p_ext, test_y_ext = test_external

    threshold = split_conformal(cal_scores, binary_nonconformity(test_p_int), alpha=alpha)["threshold"]

    def _coverage(test_p, test_y, thr):
        test_y = np.asarray(test_y).astype(int)
        scores = binary_nonconformity(test_p)
        true_scores = scores[np.arange(len(test_y)), test_y]
        return float((true_scores <= thr).mean())

    result = {
        "alpha": alpha,
        "nominal_coverage": 1 - alpha,
        "threshold": threshold,
        "internal_coverage": _coverage(test_p_int, test_y_int, threshold),
        "external_coverage": _coverage(test_p_ext, test_y_ext, threshold),
    }
    result["coverage_violated"] = bool(result["external_coverage"] < result["nominal_coverage"] - 0.02)

    if x_cal is not None and x_external is not None:
        result["weighted_mitigation"] = _weighted_conformal(
            cal_p, cal_y, x_cal, test_p_ext, test_y_ext, x_external, alpha
        )

    return result


def _weighted_conformal(cal_p, cal_y, x_cal, test_p_ext, test_y_ext, x_external, alpha):
    from sklearn.linear_model import LogisticRegression

    x_dom = np.concatenate([x_cal, x_external])
    y_dom = np.concatenate([np.zeros(len(x_cal)), np.ones(len(x_external))])
    dom_clf = LogisticRegression(max_iter=1000).fit(x_dom, y_dom)

    # Likelihood ratio external/internal at each calibration point, from the
    # domain classifier's posterior (Tibshirani et al. 2019, Sec. 2).
    p_ext_given_cal = dom_clf.predict_proba(x_cal)[:, 1]
    weights = p_ext_given_cal / np.clip(1 - p_ext_given_cal, 1e-6, None)

    cal_scores = binary_nonconformity(cal_p)[np.arange(len(cal_y)), cal_y]
    order = np.argsort(cal_scores)
    sorted_scores, sorted_w = cal_scores[order], weights[order]
    cum_w = np.cumsum(sorted_w)
    total_w = cum_w[-1] + 1.0  # +1: placeholder mass for the test point itself
    idx = min(int(np.searchsorted(cum_w, (1 - alpha) * total_w)), len(sorted_scores) - 1)
    weighted_threshold = float(sorted_scores[idx])

    test_y_ext = np.asarray(test_y_ext).astype(int)
    scores_ext = binary_nonconformity(test_p_ext)
    true_scores_ext = scores_ext[np.arange(len(test_y_ext)), test_y_ext]

    return {
        "threshold": weighted_threshold,
        "recovered_external_coverage": float((true_scores_ext <= weighted_threshold).mean()),
    }

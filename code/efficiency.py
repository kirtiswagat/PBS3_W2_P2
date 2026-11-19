"""Latency and memory benchmark for a feature subset (H2 support).

An equivalence claim ("k features are as good as all of them") needs latency
and memory numbers alongside AUC, or it is only half the argument. Fit time is
averaged over n_repeats to smooth OS scheduling noise; predict time is
reported per 1,000 rows so subsets of different sizes are comparable.
"""

import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from heads import fit_head


def benchmark_subset(x, y, w, indices, C, n_repeats=5, max_iter=2000):
    xs = x[:, indices]

    fit_times = []
    clf = None
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf = fit_head(xs, y, w, C=C, max_iter=max_iter)
        fit_times.append(time.perf_counter() - t0)

    if clf is None:
        return {
            "n_features": int(len(indices)),
            "fit_seconds": float(np.mean(fit_times)),
            "predict_ms_per_1k": float("nan"),
            "memory_mb": float(xs.nbytes / 1e6),
        }

    predict_times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        clf.predict_proba(xs)
        predict_times.append(time.perf_counter() - t0)
    ms_per_1k = np.mean(predict_times) / len(xs) * 1000 * 1000

    return {
        "n_features": int(len(indices)),
        "fit_seconds": float(np.mean(fit_times)),
        "predict_ms_per_1k": float(ms_per_1k),
        "memory_mb": float(xs.nbytes / 1e6),
    }

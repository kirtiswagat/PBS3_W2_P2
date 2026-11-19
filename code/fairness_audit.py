"""Post-terminal fairness audit on patient-disjoint Train-C data.

This audit deliberately avoids expert.npz and NIH. It uses the frozen primary
Xception/C=0.1 policies, fits heads on Train-A, and joins demographics from
the complete CheXpert metadata table so Train-C rows are not missed.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

import heads
import trunks
from fairness import chexpert_groups, equalised_odds_gap, subgroup_auc
from heads import Standardizer, apply_policy, fit_head, load_split, set_trunk
from metrics import per_class_metrics

BASE = str(Path(__file__).resolve().parent.parent)
OUT = os.path.join(BASE, "results")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trunk", default="xception", choices=list(trunks.TRUNKS))
    parser.add_argument("--C", type=float, default=0.1)
    parser.add_argument("--sweep-tag", default="sweep_train_c_xception_C0.1")
    parser.add_argument(
        "--metadata-csv",
        default=os.path.join(BASE, "..", "Dataset", "CheXpert-v1.0-small", "train.csv"),
    )
    parser.add_argument("--tag", default="fairness_train_c_xception")
    args = parser.parse_args()

    set_trunk(args.trunk)
    with open(os.path.join(OUT, f"{args.sweep_tag}.json")) as handle:
        sweep = json.load(handle)

    train = load_split("train_a")
    evalset = load_split("train_c")
    names = train["label_names"]
    scaler = Standardizer().fit(train["emb_backbone"])
    x_train = scaler.transform(train["emb_backbone"])
    x_eval = scaler.transform(evalset["emb_backbone"])
    groups = chexpert_groups(evalset["paths"], args.metadata_csv)

    results = {}
    for label, sweep_result in sweep["results"].items():
        if label not in names:
            continue
        index = names.index(label)
        policy = sweep_result.get("_best")
        threshold = sweep_result.get(policy, {}).get("threshold_mean", 0.5)
        train_idx, y_train, weights = apply_policy(train["labels"][:, index], policy)
        classifier = fit_head(
            x_train[train_idx], y_train, weights, C=args.C, max_iter=2000
        )
        if classifier is None:
            continue

        y_raw = evalset["labels"][:, index]
        valid = y_raw != -1
        y_eval = np.nan_to_num(y_raw[valid], nan=0.0)
        probabilities = classifier.predict_proba(x_eval[valid])[:, 1]
        eval_groups = [group for group, keep in zip(groups, valid) if keep]
        sex_groups = [group["sex"] if group else None for group in eval_groups]
        age_groups = [group["age_band"] if group else None for group in eval_groups]

        results[label] = {
            "policy": policy,
            "threshold": float(threshold),
            "internal_train_c": per_class_metrics(
                y_eval, probabilities, threshold=threshold
            ),
            "metadata_rows_matched": int(
                sum(group is not None for group in eval_groups)
            ),
            "auc_by_sex": subgroup_auc(y_eval, probabilities, sex_groups),
            "auc_by_age": subgroup_auc(y_eval, probabilities, age_groups),
            "equalised_odds_by_sex": equalised_odds_gap(
                y_eval, probabilities, sex_groups, threshold
            ),
            "equalised_odds_by_age": equalised_odds_gap(
                y_eval, probabilities, age_groups, threshold
            ),
        }

    payload = {
        "phase": "fairness_audit",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "trunk": args.trunk,
        "C": args.C,
        "fit_split": "train_a",
        "evaluation_split": "train_c",
        "metadata_csv": args.metadata_csv,
        "sweep_tag": args.sweep_tag,
        "expert_opened": False,
        "nih_opened": False,
        "results": results,
    }
    path = os.path.join(OUT, f"{args.tag}.json")
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    print(f"Written {path}")
    print(f"Labels audited: {len(results)}")
    for label, result in results.items():
        print(
            f"{label:<20} matched={result['metadata_rows_matched']:5d} "
            f"sex_groups={len(result['auc_by_sex'])} "
            f"age_groups={len(result['auc_by_age'])}"
        )


if __name__ == "__main__":
    main()

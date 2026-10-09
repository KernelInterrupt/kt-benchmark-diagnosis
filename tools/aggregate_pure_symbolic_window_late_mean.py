#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASETS = ["nips_task34", "algebra2005", "assist2015"]
DEFAULT_METHODS = ["contextmix"]


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate pure symbolic question-window late_mean metrics.")
    parser.add_argument("--datasets", default="nips_task34,algebra2005,assist2015")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "final_pure_symbolic_window_late_mean_20260502",
    )
    return parser.parse_args()


def parse_tokens(value):
    if value is None:
        return []
    tokens = str(value).split(",")
    if tokens and tokens[-1] == "":
        tokens = tokens[:-1]
    return tokens


def safe_int(value, default=-1):
    try:
        return int(float(value))
    except Exception:
        return default


def auc_score(y, p):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    n1 = int(y.sum())
    n0 = int(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=np.float64)
    sorted_p = p[order]
    i = 0
    rank = 1
    while i < len(p):
        j = i + 1
        while j < len(p) and sorted_p[j] == sorted_p[i]:
            j += 1
        avg_rank = (rank + rank + (j - i) - 1) / 2.0
        ranks[order[i:j]] = avg_rank
        rank += (j - i)
        i = j
    sum_ranks_pos = ranks[y == 1].sum()
    return float((sum_ranks_pos - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def fold_task(task):
    dataset, method, fold, qwin_path = task
    ys = []
    ps = []
    with Path(qwin_path).open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            qidxs = parse_tokens(row.get("qidxs"))
            orirows = parse_tokens(row.get("orirow"))
            responses = parse_tokens(row.get("responses"))
            probs = parse_tokens(row.get("ctw_pseqs"))
            selectmasks = parse_tokens(row.get("selectmasks"))
            limit = min(len(qidxs), len(orirows), len(responses), len(probs), len(selectmasks))
            for pos in range(limit):
                if safe_int(selectmasks[pos]) != 1:
                    continue
                if safe_int(qidxs[pos]) < 0 or safe_int(orirows[pos]) < 0:
                    continue
                y = safe_int(responses[pos])
                if y < 0:
                    continue
                try:
                    p = float(probs[pos])
                except Exception:
                    continue
                if p < 0:
                    continue
                ys.append(y)
                ps.append(p)
    ys_arr = np.asarray(ys, dtype=np.int64)
    ps_arr = np.asarray(ps, dtype=np.float64)
    auc = auc_score(ys_arr, ps_arr)
    acc = float(((ps_arr >= 0.5).astype(np.int64) == ys_arr).mean()) if len(ys_arr) else float("nan")
    return {
        "dataset": dataset,
        "method": method,
        "fold": int(fold),
        "windowauclate_mean": auc,
        "windowacclate_mean": acc,
        "n_rows": int(len(ys_arr)),
        "source": str(Path(qwin_path).relative_to(REPO_ROOT)),
    }


def build_tasks(datasets, methods):
    tasks = []
    base = REPO_ROOT / "runs" / "symbolic_hybrid_benchmark_parallel" / "fold_jobs"
    for dataset in datasets:
        ds_root = base / dataset
        for method in methods:
            for fold_dir in sorted(ds_root.glob(f"{method}_fold*")):
                info_path = fold_dir / method / "artifact_info.json"
                if not info_path.exists():
                    continue
                info = json.loads(info_path.read_text())
                for fold_key, files in info.get("test_by_fold", {}).items():
                    qwin = REPO_ROOT / files["test_question_window_file"]
                    if qwin.exists():
                        tasks.append((dataset, method, int(fold_key), str(qwin)))
    return tasks


def main():
    args = parse_args()
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    tasks = build_tasks(datasets, methods)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        per_fold = list(ex.map(fold_task, tasks))

    per_fold = sorted(per_fold, key=lambda r: (r["dataset"], r["method"], r["fold"]))
    pd.DataFrame(per_fold).to_csv(args.out_dir / "per_fold_window_late_mean.csv", index=False)

    summary_rows = []
    for (dataset, method), group in pd.DataFrame(per_fold).groupby(["dataset", "method"], sort=True):
        aucs = group["windowauclate_mean"].astype(float).to_numpy()
        accs = group["windowacclate_mean"].astype(float).to_numpy()
        summary_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "num_folds": int(len(group)),
                "windowauclate_mean_mean": float(np.mean(aucs)),
                "windowauclate_mean_std": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
                "windowacclate_mean_mean": float(np.mean(accs)),
                "windowacclate_mean_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
                "folds": ",".join(str(int(x)) for x in group["fold"].tolist()),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(["dataset", "method"])
    summary_df.to_csv(args.out_dir / "summary_window_late_mean.csv", index=False)

    print(f"WROTE {args.out_dir / 'per_fold_window_late_mean.csv'}")
    print(f"WROTE {args.out_dir / 'summary_window_late_mean.csv'}")
    for row in summary_df.to_dict("records"):
        print(
            f"{row['dataset']}\t{row['method']}\t"
            f"AUC={row['windowauclate_mean_mean']:.6f}±{row['windowauclate_mean_std']:.6f}\t"
            f"ACC={row['windowacclate_mean_mean']:.6f}±{row['windowacclate_mean_std']:.6f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())

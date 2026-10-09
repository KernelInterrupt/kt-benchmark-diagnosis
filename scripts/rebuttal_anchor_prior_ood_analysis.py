#!/usr/bin/env python3
"""Fast anchor sensitivity and item-support OOD checks for rebuttal evidence.

This script reuses the exact common-sample parquets produced by
`rebuttal_unified_analysis.py`. It does not retrain neural KT models and does
not rebuild the model-output intersection.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rebuttal_unified_analysis import (  # noqa: E402
    AXES,
    DATASETS,
    MODEL_COLUMNS,
    TRAIN_FILES,
    basic_metrics,
    binary_entropy_bits,
    fixed_bands,
    item_parameters,
    load_item_counts,
    online_rasch_predictions,
)


METRICS = ("acc", "brier", "logloss")
DEFAULT_MODEL_ORDER = ("CTW", "ItemMean", "ItemRasch", "CTW+NN", "AKT", "simpleKT", "stableKT")


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument(
        "--common-dir",
        type=Path,
        default=Path("runs/rebuttal_unified_item_anchor_20260725/common_samples"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/rebuttal_anchor_sensitivity_ood_20260726"),
    )
    parser.add_argument("--prior-strengths", default="1,5,20,100,500")
    parser.add_argument("--rasch-prior-precisions", default="0.25,0.5,1,2,4")
    return parser.parse_args()


def summarize_five_folds(frame: pd.DataFrame, group_columns: list[str], value_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_columns, dropna=False, observed=False):
        row = dict(zip(group_columns, keys))
        if "n" in group:
            row["n_total"] = int(group["n"].sum())
        row["n_folds"] = int(group["fold"].nunique()) if "fold" in group else int(len(group))
        for column in value_columns:
            values = group[column].astype(float).dropna()
            if values.empty:
                continue
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            se = std / math.sqrt(len(values)) if len(values) > 0 else math.nan
            halfwidth = float(stats.t.ppf(0.975, len(values) - 1) * se) if len(values) > 1 else 0.0
            row[f"{column}_mean"] = mean
            row[f"{column}_std"] = std
            row[f"{column}_se"] = se
            row[f"{column}_ci95_halfwidth"] = halfwidth
            row[f"{column}_ci95_low"] = mean - halfwidth
            row[f"{column}_ci95_high"] = mean + halfwidth
        rows.append(row)
    return pd.DataFrame(rows)


def add_metric_rows(
    rows: list[dict[str, object]],
    *,
    dataset: str,
    fold: int,
    scope: str,
    frame: pd.DataFrame,
    item_prior_strength: float | None = None,
    rasch_prior_precision: float | None = None,
    slice_label: str | None = None,
) -> None:
    if frame.empty:
        return
    y = frame["y_true"].to_numpy(dtype=int)
    reference_ctw = basic_metrics(y, frame["p_ctw"].to_numpy(dtype=float))
    reference_rasch = (
        basic_metrics(y, frame["p_item_rasch"].to_numpy(dtype=float))
        if "p_item_rasch" in frame.columns
        else None
    )
    for model in DEFAULT_MODEL_ORDER:
        probability_column = MODEL_COLUMNS[model]
        if probability_column not in frame.columns:
            continue
        values = basic_metrics(y, frame[probability_column].to_numpy(dtype=float))
        row: dict[str, object] = {
            "dataset": dataset,
            "fold": fold,
            "scope": scope,
            "model": model,
            "n": int(len(frame)),
            **{metric: values[metric] for metric in METRICS},
        }
        if item_prior_strength is not None:
            row["item_prior_strength"] = item_prior_strength
        if rasch_prior_precision is not None:
            row["rasch_prior_precision"] = rasch_prior_precision
        if slice_label is not None:
            row["slice"] = slice_label
        for metric in METRICS:
            row[f"delta_{metric}_vs_CTW"] = values[metric] - reference_ctw[metric]
            if reference_rasch is not None:
                row[f"delta_{metric}_vs_ItemRasch"] = values[metric] - reference_rasch[metric]
        rows.append(row)


def load_common_fold(common_dir: Path, dataset: str, fold: int, columns: list[str] | None = None) -> pd.DataFrame:
    path = common_dir / f"{dataset}_fold{fold}.parquet"
    frame = pd.read_parquet(path, columns=columns)
    return frame.sort_values(["orirow", "qidx"]).reset_index(drop=True)


def run_prior_sensitivity(
    repo: Path,
    common_dir: Path,
    prior_strengths: list[float],
    prior_precisions: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    columns = [
        "fold",
        "orirow",
        "qidx",
        "y_true",
        "item_id",
        "p_ctw",
        "p_ctw_nn",
        "p_akt",
        "p_simplekt",
        "p_stablekt",
        "dataset",
    ]

    for dataset in DATASETS:
        by_fold, global_counts, interaction_counts = load_item_counts(repo / TRAIN_FILES[dataset])
        for fold in range(5):
            base = load_common_fold(common_dir, dataset, fold, columns=columns)
            for prior_strength in prior_strengths:
                item_probs, global_rate, supports = item_parameters(fold, by_fold, global_counts, prior_strength)
                frame = base.copy()
                frame["p_item_mean"] = frame["item_id"].map(item_probs).fillna(global_rate).astype(float)
                frame["item_train_support"] = frame["item_id"].map(supports).fillna(0).astype(int)
                training_rows.append(
                    {
                        "dataset": dataset,
                        "fold": fold,
                        "item_prior_strength": prior_strength,
                        "training_interactions": sum(interaction_counts.values()) - interaction_counts[fold],
                        "training_items": sum(1 for support in supports.values() if support > 0),
                        "global_correctness": global_rate,
                    }
                )
                for prior_precision in prior_precisions:
                    local = frame.copy()
                    local["p_item_rasch"] = online_rasch_predictions(local, prior_precision)
                    entropy = binary_entropy_bits(local["p_item_rasch"].to_numpy(dtype=float))
                    bands = fixed_bands(entropy)
                    add_metric_rows(
                        rows,
                        dataset=dataset,
                        fold=fold,
                        scope="global",
                        frame=local,
                        item_prior_strength=prior_strength,
                        rasch_prior_precision=prior_precision,
                    )
                    for band in ("0.0-0.2", "0.8-1.0"):
                        mask = np.asarray(bands == band)
                        add_metric_rows(
                            rows,
                            dataset=dataset,
                            fold=fold,
                            scope=f"item_rasch_entropy_{band}",
                            frame=local.loc[mask],
                            item_prior_strength=prior_strength,
                            rasch_prior_precision=prior_precision,
                        )

    by_fold = pd.DataFrame(rows)
    value_columns = [column for column in by_fold.columns if column.startswith("delta_")] + list(METRICS)
    summary = summarize_five_folds(
        by_fold,
        ["dataset", "item_prior_strength", "rasch_prior_precision", "scope", "model"],
        value_columns,
    )
    training = pd.DataFrame(training_rows)
    return by_fold, summary, training


def support_slices(frame: pd.DataFrame) -> dict[str, pd.Series]:
    support = frame["item_train_support"].astype(float)
    slices: dict[str, pd.Series] = {
        "full_common": pd.Series(True, index=frame.index),
        "zero_item_support": support == 0,
        "seen_item_support": support > 0,
    }

    ranked = support.rank(method="first")
    support_quintiles = pd.qcut(ranked, q=5, labels=["Q1_lowest", "Q2", "Q3", "Q4", "Q5_highest"])
    slices["support_q1_lowest_20pct"] = pd.Series(support_quintiles == "Q1_lowest", index=frame.index)
    slices["support_q5_highest_20pct"] = pd.Series(support_quintiles == "Q5_highest", index=frame.index)

    seen = support > 0
    if int(seen.sum()) >= 5:
        seen_ranked = support[seen].rank(method="first")
        seen_quintiles = pd.qcut(seen_ranked, q=5, labels=["Q1_seen_lowest", "Q2", "Q3", "Q4", "Q5_seen_highest"])
        low_seen = pd.Series(False, index=frame.index)
        high_seen = pd.Series(False, index=frame.index)
        low_seen.loc[seen] = np.asarray(seen_quintiles == "Q1_seen_lowest")
        high_seen.loc[seen] = np.asarray(seen_quintiles == "Q5_seen_highest")
        slices["seen_support_q1_lowest_20pct"] = low_seen
        slices["seen_support_q5_highest_20pct"] = high_seen

    ctw_high = pd.Series(
        np.asarray(fixed_bands(binary_entropy_bits(frame["p_ctw"].to_numpy(dtype=float))) == "0.8-1.0"),
        index=frame.index,
    )
    rasch_high = pd.Series(
        np.asarray(fixed_bands(binary_entropy_bits(frame["p_item_rasch"].to_numpy(dtype=float))) == "0.8-1.0"),
        index=frame.index,
    )
    slices["ctw_high_entropy"] = ctw_high
    slices["item_rasch_high_entropy"] = rasch_high
    slices["ctw_high_entropy_zero_item_support"] = ctw_high & (support == 0)
    slices["ctw_high_entropy_support_q1"] = ctw_high & slices["support_q1_lowest_20pct"]
    slices["item_rasch_high_entropy_zero_item_support"] = rasch_high & (support == 0)
    slices["item_rasch_high_entropy_support_q1"] = rasch_high & slices["support_q1_lowest_20pct"]
    return slices


def run_support_ood(common_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    columns = [
        "fold",
        "orirow",
        "qidx",
        "y_true",
        "item_id",
        "item_train_support",
        "p_ctw",
        "p_ctw_nn",
        "p_item_mean",
        "p_item_rasch",
        "p_akt",
        "p_simplekt",
        "p_stablekt",
        "dataset",
    ]

    for dataset in DATASETS:
        for fold in range(5):
            frame = load_common_fold(common_dir, dataset, fold, columns=columns)
            for label, mask in support_slices(frame).items():
                if int(mask.sum()) == 0:
                    continue
                add_metric_rows(
                    rows,
                    dataset=dataset,
                    fold=fold,
                    scope="item_support_ood_slice",
                    frame=frame.loc[mask],
                    slice_label=label,
                )

    by_fold = pd.DataFrame(rows)
    value_columns = [column for column in by_fold.columns if column.startswith("delta_")] + list(METRICS)
    summary = summarize_five_folds(by_fold, ["dataset", "slice", "model"], value_columns)
    return by_fold, summary


def build_existing_transfer_index(repo: Path) -> pd.DataFrame:
    candidates = [
        (
            "assist2009_question_heldout_neural_only",
            repo / "runs/assist2009_cold_nn_only_20260504/cold_nn_only_ensemble_summary.csv",
            "cold question-heldout neural-only and post-hoc ensemble metrics",
        ),
        (
            "assist2009_question_heldout_residual",
            repo / "runs/assist2009_cold_final_eval_20260503/selected_residual_summary_by_backend_expert.csv",
            "cold question-heldout residual KT summaries by symbolic backend",
        ),
        (
            "multi_dataset_global_convex",
            repo / "runs/all_datasets_global_convex_summary_20260504/best_global_convex_rows.csv",
            "best existing global convex rows across NIPS/Algebra/ASSIST/Bridge artifacts",
        ),
        (
            "assist2009_contextmix_validation_fitted",
            repo / "runs/assist2009_contextmix_validation_fitted_residual_ensemble_20260725/summary.csv",
            "ASSIST2009 contextmix validation-fitted residual ensemble metrics",
        ),
    ]
    rows: list[dict[str, object]] = []
    for name, path, description in candidates:
        row: dict[str, object] = {
            "artifact": name,
            "path": str(path.relative_to(repo)) if path.exists() else str(path),
            "exists": path.exists(),
            "description": description,
        }
        if path.exists():
            try:
                frame = pd.read_csv(path)
                row["n_rows"] = len(frame)
                row["columns"] = ",".join(frame.columns.astype(str))
            except Exception as exc:  # pragma: no cover - diagnostic only.
                row["read_error"] = str(exc)
        rows.append(row)
    return pd.DataFrame(rows)


def write_readme(output: Path, prior_strengths: list[float], prior_precisions: list[float]) -> None:
    content = f"""# Anchor Sensitivity and Item-Support OOD Checks

This directory contains lightweight rebuttal-side experiments. No neural KT
model is retrained. All new row-level metrics reuse the exact common sample
from `runs/rebuttal_unified_item_anchor_20260725/common_samples`.

## Prior sensitivity

Grid:

- item empirical-Bayes prior strength: `{prior_strengths}`
- online Rasch prior precision: `{prior_precisions}`

Files:

- `prior_sensitivity_by_fold.csv`: fold-level global, low-entropy, and
  high-entropy metrics for every grid point.
- `prior_sensitivity_summary_ci.csv`: five-fold mean, SE, and 95% t intervals.
- `prior_sensitivity_key_results.csv`: CTW+NN-SKT minus ItemRasch in the
  highest ItemRasch-entropy band for every grid point.
- `item_prior_training_summary_by_grid.csv`: fold-level item-anchor training
  coverage for every item-prior strength.

## OOD/support slices

The support-slice check evaluates the default ItemMean/ItemRasch anchor
(`prior_strength=20`, `prior_precision=1`) under item-statistics stress:
zero training support, low-support quintiles, seen-item support, and their
intersection with high CTW or high ItemRasch entropy.

Files:

- `support_ood_metrics_by_fold.csv`: fold-level metrics by support slice.
- `support_ood_summary_ci.csv`: five-fold summaries by support slice.
- `support_ood_key_results.csv`: CTW+NN-SKT deltas against CTW and ItemRasch
  on the most relevant support/OOD slices.

## Existing transfer/cold artifacts

`existing_transfer_artifact_index.csv` lists local cold/transfer-adjacent
artifacts found in the repository. They are intentionally kept separate from
the unified NIPS/Algebra common-sample experiments because their protocols and
row contracts differ.
"""
    (output / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    common_dir = args.common_dir if args.common_dir.is_absolute() else repo / args.common_dir
    output = args.output if args.output.is_absolute() else repo / args.output
    output.mkdir(parents=True, exist_ok=True)

    prior_strengths = parse_float_list(args.prior_strengths)
    prior_precisions = parse_float_list(args.rasch_prior_precisions)

    prior_by_fold, prior_summary, training = run_prior_sensitivity(
        repo, common_dir, prior_strengths, prior_precisions
    )
    prior_by_fold.to_csv(output / "prior_sensitivity_by_fold.csv", index=False)
    prior_summary.to_csv(output / "prior_sensitivity_summary_ci.csv", index=False)
    training.to_csv(output / "item_prior_training_summary_by_grid.csv", index=False)

    prior_key = prior_summary[
        (prior_summary["scope"] == "item_rasch_entropy_0.8-1.0")
        & (prior_summary["model"] == "CTW+NN")
    ].copy()
    prior_key = prior_key.sort_values(["dataset", "item_prior_strength", "rasch_prior_precision"])
    prior_key.to_csv(output / "prior_sensitivity_key_results.csv", index=False)

    support_by_fold, support_summary = run_support_ood(common_dir)
    support_by_fold.to_csv(output / "support_ood_metrics_by_fold.csv", index=False)
    support_summary.to_csv(output / "support_ood_summary_ci.csv", index=False)
    support_key = support_summary[support_summary["model"] == "CTW+NN"].copy()
    support_key = support_key[
        support_key["slice"].isin(
            [
                "full_common",
                "zero_item_support",
                "seen_item_support",
                "support_q1_lowest_20pct",
                "seen_support_q1_lowest_20pct",
                "ctw_high_entropy_zero_item_support",
                "ctw_high_entropy_support_q1",
                "item_rasch_high_entropy_zero_item_support",
                "item_rasch_high_entropy_support_q1",
            ]
        )
    ]
    support_key = support_key.sort_values(["dataset", "slice"])
    support_key.to_csv(output / "support_ood_key_results.csv", index=False)

    build_existing_transfer_index(repo).to_csv(output / "existing_transfer_artifact_index.csv", index=False)

    metadata = {
        "common_dir": str(common_dir),
        "prior_strengths": prior_strengths,
        "rasch_prior_precisions": prior_precisions,
        "default_support_slice_anchor": {"item_prior_strength": 20.0, "rasch_prior_precision": 1.0},
        "datasets": list(DATASETS),
        "metric_columns": list(METRICS),
        "model_columns": MODEL_COLUMNS,
        "axes": AXES,
    }
    (output / "analysis_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_readme(output, prior_strengths, prior_precisions)
    print(f"Wrote anchor sensitivity/OOD outputs to {output}")


if __name__ == "__main__":
    main()

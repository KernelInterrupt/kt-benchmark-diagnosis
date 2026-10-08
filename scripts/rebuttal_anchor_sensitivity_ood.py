#!/usr/bin/env python3
"""Run quick robustness checks for the rebuttal item-aware anchor.

This script reuses the exact common samples produced by
``rebuttal_unified_analysis.py``. It does not retrain neural models and does
not change the existing unified outputs. The checks are intentionally scoped to
fast post-processing experiments:

1. Item prior-strength and online Rasch prior-precision sensitivity.
2. Low/zero item-support stress tests on the same common prediction rows.
3. A small index of already available cold/transfer-style artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from rebuttal_unified_analysis import (
    DATASETS,
    EPS,
    MODEL_COLUMNS,
    TRAIN_FILES,
    basic_metrics,
    binary_entropy_bits,
    calibration_metrics,
    fixed_bands,
    item_parameters,
    load_item_counts,
    normalize_item,
)


CORE_EVAL_MODELS = ("CTW", "CTW+NN", "AKT", "simpleKT", "stableKT")
DEFAULT_PRIOR_STRENGTH = 20.0
DEFAULT_RASCH_PRECISION = 1.0


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


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
    parser.add_argument("--prior-strengths", default="5,20,100")
    parser.add_argument("--rasch-precisions", default="0.25,1,4")
    parser.add_argument("--ece-bins", type=int, default=15)
    return parser.parse_args()


def ci_summary(values: pd.Series) -> dict[str, float]:
    values = values.astype(float)
    n = int(values.shape[0])
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 0 else math.nan
    halfwidth = float(stats.t.ppf(0.975, n - 1) * se) if n > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "se": se,
        "ci95_halfwidth": halfwidth,
        "ci95_low": mean - halfwidth,
        "ci95_high": mean + halfwidth,
    }


def summarize_by_fold(frame: pd.DataFrame, group_columns: list[str], value_columns: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_columns, dropna=False, observed=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["n_total"] = int(group["n"].sum()) if "n" in group else int(len(group))
        row["n_folds"] = int(group["fold"].nunique()) if "fold" in group else int(len(group))
        for column in value_columns:
            stats_row = ci_summary(group[column])
            for suffix, value in stats_row.items():
                row[f"{column}_{suffix}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def with_item_anchor(
    frame: pd.DataFrame,
    item_probs: dict[str, float],
    global_rate: float,
    prior_precision: float,
) -> pd.DataFrame:
    local = frame.copy()
    local["p_item_mean_sens"] = local["item_id"].map(item_probs).fillna(global_rate).astype(float)
    local["p_item_mean"] = local["p_item_mean_sens"]
    local["p_item_rasch_sens"] = fast_online_rasch_predictions(local, prior_precision)
    return local


def fast_online_rasch_predictions(frame: pd.DataFrame, prior_precision: float) -> np.ndarray:
    """Equivalent to rebuttal_unified_analysis.online_rasch_predictions, but avoids pandas scalar reads."""
    p_item = np.clip(frame["p_item_mean"].to_numpy(dtype=float), EPS, 1.0 - EPS)
    base_logits = np.log(p_item / (1.0 - p_item))
    y = frame["y_true"].to_numpy(dtype=np.int8)
    orirow = frame["orirow"].to_numpy()
    result = np.empty(len(frame), dtype=float)
    positions = np.empty(len(frame), dtype=np.int32)

    n = len(frame)
    start = 0
    while start < n:
        end = start + 1
        current = orirow[start]
        while end < n and orirow[end] == current:
            end += 1

        theta = 0.0
        precision = float(prior_precision)
        for offset, idx in enumerate(range(start, end), start=1):
            score = base_logits[idx] + theta
            if score > 30.0:
                score = 30.0
            elif score < -30.0:
                score = -30.0
            p = 1.0 / (1.0 + math.exp(-score))
            result[idx] = p
            positions[idx] = offset
            precision += max(p * (1.0 - p), 1e-6)
            theta += (float(y[idx]) - p) / precision
            if theta > 4.0:
                theta = 4.0
            elif theta < -4.0:
                theta = -4.0

        start = end

    frame["seq_pos"] = positions
    return result


def metric_row(
    frame: pd.DataFrame,
    dataset: str,
    fold: int,
    model: str,
    probability_column: str,
    reference_values: dict[str, float] | None,
    ece_bins: int,
    extra: dict[str, object],
) -> dict[str, object]:
    y = frame["y_true"].to_numpy(dtype=int)
    p = frame[probability_column].to_numpy(dtype=float)
    values = basic_metrics(y, p)
    values.update(calibration_metrics(y, p, ece_bins)[0])
    row: dict[str, object] = {
        "dataset": dataset,
        "fold": fold,
        "model": model,
        "n": int(len(frame)),
        **extra,
        **values,
    }
    if reference_values is not None:
        for metric in ("acc", "brier", "logloss", "ece", "mce", "signed_calibration_gap"):
            row[f"delta_{metric}_vs_anchor"] = values[metric] - reference_values[metric]
    return row


def support_slices(frame: pd.DataFrame) -> dict[str, pd.Series]:
    support = frame["item_train_support"].astype(float)
    positive = support[support > 0]
    low_positive_cutoff = float(positive.quantile(0.20)) if not positive.empty else math.nan
    high_positive_cutoff = float(positive.quantile(0.80)) if not positive.empty else math.nan
    return {
        "all": pd.Series(True, index=frame.index),
        "zero_item_support": support == 0,
        "seen_item_support": support > 0,
        "seen_low_support_q20": (support > 0) & (support <= low_positive_cutoff),
        "seen_high_support_q80": (support > 0) & (support >= high_positive_cutoff),
    }


def add_sensitivity_rows(
    frame: pd.DataFrame,
    dataset: str,
    fold: int,
    prior_strength: float,
    prior_precision: float,
    ece_bins: int,
    rows_global: list[dict[str, object]],
    rows_high: list[dict[str, object]],
) -> None:
    global_reference = basic_metrics(
        frame["y_true"].to_numpy(dtype=int),
        frame["p_item_rasch_sens"].to_numpy(dtype=float),
    )
    global_reference.update(
        calibration_metrics(
            frame["y_true"].to_numpy(dtype=int),
            frame["p_item_rasch_sens"].to_numpy(dtype=float),
            ece_bins,
        )[0]
    )
    extra = {
        "prior_strength": prior_strength,
        "rasch_prior_precision": prior_precision,
        "reference_model": "ItemRasch",
        "scope": "global",
    }
    for model, column in {
        "ItemMean": "p_item_mean_sens",
        "ItemRasch": "p_item_rasch_sens",
        **{m: MODEL_COLUMNS[m] for m in CORE_EVAL_MODELS},
    }.items():
        reference = None if model == "ItemRasch" else global_reference
        rows_global.append(metric_row(frame, dataset, fold, model, column, reference, ece_bins, extra))

    entropy = binary_entropy_bits(frame["p_item_rasch_sens"].to_numpy(dtype=float))
    bands = fixed_bands(entropy)
    high = frame[np.asarray(bands == "0.8-1.0")].copy()
    if high.empty:
        return
    y_high = high["y_true"].to_numpy(dtype=int)
    high_reference = basic_metrics(y_high, high["p_item_rasch_sens"].to_numpy(dtype=float))
    high_reference.update(calibration_metrics(y_high, high["p_item_rasch_sens"].to_numpy(dtype=float), ece_bins)[0])
    extra_high = {
        "prior_strength": prior_strength,
        "rasch_prior_precision": prior_precision,
        "reference_model": "ItemRasch",
        "axis": "item_rasch",
        "scheme": "fixed_width",
        "band": "0.8-1.0",
        "high_band_share": float(len(high) / len(frame)),
    }
    for model, column in {m: MODEL_COLUMNS[m] for m in CORE_EVAL_MODELS}.items():
        rows_high.append(metric_row(high, dataset, fold, model, column, high_reference, ece_bins, extra_high))


def add_support_slice_rows(
    frame: pd.DataFrame,
    dataset: str,
    fold: int,
    prior_strength: float,
    prior_precision: float,
    ece_bins: int,
    rows: list[dict[str, object]],
    count_rows: list[dict[str, object]],
) -> None:
    anchored_columns = {
        "ctw": ("p_ctw", "CTW"),
        "item_rasch": ("p_item_rasch_sens", "ItemRasch"),
    }
    for slice_name, mask in support_slices(frame).items():
        local = frame[mask].copy()
        count_rows.append(
            {
                "dataset": dataset,
                "fold": fold,
                "slice": slice_name,
                "n": int(len(local)),
                "share_of_fold": float(len(local) / len(frame)) if len(frame) else math.nan,
                "item_support_min": float(local["item_train_support"].min()) if len(local) else math.nan,
                "item_support_median": float(local["item_train_support"].median()) if len(local) else math.nan,
                "item_support_max": float(local["item_train_support"].max()) if len(local) else math.nan,
            }
        )
        if len(local) == 0:
            continue
        for axis, (axis_column, reference_model) in anchored_columns.items():
            entropy = binary_entropy_bits(local[axis_column].to_numpy(dtype=float))
            bands = fixed_bands(entropy)
            high = local[np.asarray(bands == "0.8-1.0")].copy()
            if high.empty:
                continue
            reference_column = axis_column
            y_high = high["y_true"].to_numpy(dtype=int)
            reference_values = basic_metrics(y_high, high[reference_column].to_numpy(dtype=float))
            reference_values.update(calibration_metrics(y_high, high[reference_column].to_numpy(dtype=float), ece_bins)[0])
            extra = {
                "prior_strength": prior_strength,
                "rasch_prior_precision": prior_precision,
                "slice": slice_name,
                "axis": axis,
                "reference_model": reference_model,
                "scheme": "fixed_width",
                "band": "0.8-1.0",
                "slice_n": int(len(local)),
                "high_band_share_within_slice": float(len(high) / len(local)),
            }
            for model, column in {m: MODEL_COLUMNS[m] for m in CORE_EVAL_MODELS}.items():
                rows.append(metric_row(high, dataset, fold, model, column, reference_values, ece_bins, extra))


def write_existing_artifact_index(repo: Path, output: Path) -> None:
    candidates = [
        {
            "artifact": "assist2009 question-heldout cold NN-only ensemble",
            "path": "runs/assist2009_cold_nn_only_20260504/cold_nn_only_ensemble_summary.csv",
            "scope": "existing cold-start/OOD artifact; separate from NIPS/Algebra common sample",
        },
        {
            "artifact": "assist2009 question-heldout residual selected summary",
            "path": "runs/assist2009_cold_final_eval_20260503/selected_residual_summary_by_expert.csv",
            "scope": "existing cold-start/OOD artifact; separate from NIPS/Algebra common sample",
        },
        {
            "artifact": "all-dataset global convex summary",
            "path": "runs/all_datasets_global_convex_summary_20260504/best_global_convex_rows.csv",
            "scope": "existing multi-dataset standard evaluation artifact",
        },
        {
            "scope": "existing Bridge2006 transfer-style dataset artifact",
        },
        {
            "artifact": "assist2009 validation-fitted residual ensemble",
            "path": "runs/assist2009_contextmix_validation_fitted_residual_ensemble_20260725/summary.csv",
            "scope": "existing held-out validation-fitted ensemble artifact",
        },
        {
            "artifact": "bridge2006 validation-fitted residual ensemble",
            "scope": "existing held-out validation-fitted ensemble artifact",
        },
    ]
    rows = []
    for candidate in candidates:
        path = repo / candidate["path"]
        row = dict(candidate)
        row["exists"] = path.exists()
        if path.exists():
            try:
                table = pd.read_csv(path)
                row["n_rows"] = int(len(table))
                row["columns"] = ",".join(table.columns.astype(str).tolist())
            except Exception as exc:  # pragma: no cover - artifact audit should not fail the experiment.
                row["n_rows"] = math.nan
                row["columns"] = f"read_error:{exc}"
        rows.append(row)
    pd.DataFrame(rows).to_csv(output / "existing_ood_transfer_artifact_index.csv", index=False)


def write_prior_transfer_summary(high_by_fold: pd.DataFrame, global_by_fold: pd.DataFrame, output: Path) -> None:
    """Select anchor priors by source-dataset ItemRasch NLL and evaluate target headroom."""
    global_summary = summarize_by_fold(
        global_by_fold[global_by_fold["model"] == "ItemRasch"],
        ["dataset", "prior_strength", "rasch_prior_precision", "model"],
        ["logloss", "brier", "acc"],
    )
    high_summary = summarize_by_fold(
        high_by_fold[high_by_fold["model"] == "CTW+NN"],
        ["dataset", "prior_strength", "rasch_prior_precision", "model"],
        [
            "high_band_share",
            "delta_acc_vs_anchor",
            "delta_brier_vs_anchor",
            "delta_logloss_vs_anchor",
            "delta_ece_vs_anchor",
        ],
    )
    rows: list[dict[str, object]] = []
    for source in DATASETS:
        source_rows = global_summary[global_summary["dataset"] == source]
        if source_rows.empty:
            continue
        selected = source_rows.sort_values(["logloss_mean", "brier_mean"]).iloc[0]
        prior_strength = float(selected["prior_strength"])
        precision = float(selected["rasch_prior_precision"])
        for target in DATASETS:
            target_high = high_summary[
                (high_summary["dataset"] == target)
                & (high_summary["prior_strength"] == prior_strength)
                & (high_summary["rasch_prior_precision"] == precision)
            ]
            target_global = global_summary[
                (global_summary["dataset"] == target)
                & (global_summary["prior_strength"] == prior_strength)
                & (global_summary["rasch_prior_precision"] == precision)
            ]
            if target_high.empty or target_global.empty:
                continue
            high = target_high.iloc[0]
            global_row = target_global.iloc[0]
            rows.append(
                {
                    "source_dataset": source,
                    "target_dataset": target,
                    "selected_prior_strength": prior_strength,
                    "selected_rasch_prior_precision": precision,
                    "source_item_rasch_logloss_mean": float(selected["logloss_mean"]),
                    "target_item_rasch_logloss_mean": float(global_row["logloss_mean"]),
                    "target_high_band_share_mean": float(high["high_band_share_mean"]),
                    "target_high_delta_acc_vs_item_rasch_mean": float(high["delta_acc_vs_anchor_mean"]),
                    "target_high_delta_brier_vs_item_rasch_mean": float(high["delta_brier_vs_anchor_mean"]),
                    "target_high_delta_logloss_vs_item_rasch_mean": float(high["delta_logloss_vs_anchor_mean"]),
                    "target_high_delta_ece_vs_item_rasch_mean": float(high["delta_ece_vs_anchor_mean"]),
                }
            )
    pd.DataFrame(rows).to_csv(output / "prior_hyperparameter_transfer_summary.csv", index=False)


def write_readme(output: Path, config: dict[str, object]) -> None:
    content = f"""# Anchor Prior Sensitivity And OOD Slice Checks

This directory contains fast post-processing experiments only. No neural model
was retrained, and all NIPS/Algebra metrics reuse the exact common sample from
`{config['common_dir']}`.

## Experiments

- `prior_sensitivity_global_by_fold.csv`: global metrics for ItemMean,
  ItemRasch, CTW, CTW+NN, AKT, simpleKT, and stableKT under each item prior and
  Rasch precision setting.
- `prior_sensitivity_high_band_by_fold.csv`: highest fixed-width ItemRasch
  entropy band metrics and deltas versus ItemRasch under each setting.
- `support_slice_high_band_by_fold.csv`: default-setting high-band metrics on
  item-support stress slices (`zero_item_support`, `seen_low_support_q20`, etc.).
- `support_slice_counts_by_fold.csv`: support and share for each slice.
- `*_summary_ci.csv`: five-fold mean/SE/95% t intervals.
- `prior_sensitivity_stability_summary.csv`: min/max across the full parameter
  grid, useful for checking whether the sign and scale of the high-band
  headroom depend on one chosen prior.
- `prior_hyperparameter_transfer_summary.csv`: chooses the ItemRasch prior
  setting by global Log-loss on one dataset and reports the target-dataset
  high-band CTW+NN-vs-ItemRasch deltas under that transferred setting.
- `existing_ood_transfer_artifact_index.csv`: index of existing cold/OOD or
  multi-dataset artifacts found locally. These are not merged into the unified
  common-sample metrics because their protocols differ.

## Parameter Grid

- ItemMean prior strengths: `{config['prior_strengths']}`.
- ItemRasch prior precisions: `{config['rasch_precisions']}`.
- Default support-slice setting: prior strength `{DEFAULT_PRIOR_STRENGTH:g}`,
  Rasch precision `{DEFAULT_RASCH_PRECISION:g}`.

All ECE values use `{config['ece_bins']}` fixed probability bins inside the
reported slice/band.
"""
    (output / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    common_dir = args.common_dir if args.common_dir.is_absolute() else repo / args.common_dir
    output = args.output if args.output.is_absolute() else repo / args.output
    output.mkdir(parents=True, exist_ok=True)

    prior_strengths = parse_float_list(args.prior_strengths)
    rasch_precisions = parse_float_list(args.rasch_precisions)

    rows_global: list[dict[str, object]] = []
    rows_high: list[dict[str, object]] = []
    rows_support: list[dict[str, object]] = []
    rows_support_counts: list[dict[str, object]] = []

    for dataset in DATASETS:
        by_fold, global_counts, _ = load_item_counts(repo / TRAIN_FILES[dataset])
        item_cache: dict[tuple[int, float], tuple[dict[str, float], float]] = {}
        for fold in range(5):
            path = common_dir / f"{dataset}_fold{fold}.parquet"
            base = pd.read_parquet(path)
            base["item_id"] = base["item_id"].map(normalize_item)
            for prior_strength in prior_strengths:
                item_probs, global_rate, _ = item_parameters(fold, by_fold, global_counts, prior_strength)
                item_cache[(fold, prior_strength)] = (item_probs, global_rate)
                for precision in rasch_precisions:
                    local = with_item_anchor(base, item_probs, global_rate, precision)
                    add_sensitivity_rows(
                        local,
                        dataset,
                        fold,
                        prior_strength,
                        precision,
                        args.ece_bins,
                        rows_global,
                        rows_high,
                    )

            default_key = (fold, DEFAULT_PRIOR_STRENGTH)
            if default_key in item_cache:
                default_item_probs, default_global_rate = item_cache[default_key]
            else:
                default_item_probs, default_global_rate, _ = item_parameters(
                    fold, by_fold, global_counts, DEFAULT_PRIOR_STRENGTH
                )
            default_local = with_item_anchor(
                base, default_item_probs, default_global_rate, DEFAULT_RASCH_PRECISION
            )
            add_support_slice_rows(
                default_local,
                dataset,
                fold,
                DEFAULT_PRIOR_STRENGTH,
                DEFAULT_RASCH_PRECISION,
                args.ece_bins,
                rows_support,
                rows_support_counts,
            )

    global_by_fold = pd.DataFrame(rows_global)
    high_by_fold = pd.DataFrame(rows_high)
    support_by_fold = pd.DataFrame(rows_support)
    support_counts = pd.DataFrame(rows_support_counts)

    global_by_fold.to_csv(output / "prior_sensitivity_global_by_fold.csv", index=False)
    high_by_fold.to_csv(output / "prior_sensitivity_high_band_by_fold.csv", index=False)
    support_by_fold.to_csv(output / "support_slice_high_band_by_fold.csv", index=False)
    support_counts.to_csv(output / "support_slice_counts_by_fold.csv", index=False)

    value_columns = [
        "acc",
        "brier",
        "logloss",
        "ece",
        "delta_acc_vs_anchor",
        "delta_brier_vs_anchor",
        "delta_logloss_vs_anchor",
        "delta_ece_vs_anchor",
    ]
    summarize_by_fold(
        global_by_fold,
        ["dataset", "prior_strength", "rasch_prior_precision", "reference_model", "scope", "model"],
        [column for column in value_columns if column in global_by_fold.columns],
    ).to_csv(output / "prior_sensitivity_global_summary_ci.csv", index=False)
    summarize_by_fold(
        high_by_fold,
        [
            "dataset",
            "prior_strength",
            "rasch_prior_precision",
            "axis",
            "reference_model",
            "scheme",
            "band",
            "model",
        ],
        [column for column in ["high_band_share", *value_columns] if column in high_by_fold.columns],
    ).to_csv(output / "prior_sensitivity_high_band_summary_ci.csv", index=False)
    summarize_by_fold(
        support_by_fold,
        ["dataset", "slice", "axis", "reference_model", "scheme", "band", "model"],
        [column for column in ["high_band_share_within_slice", *value_columns] if column in support_by_fold.columns],
    ).to_csv(output / "support_slice_high_band_summary_ci.csv", index=False)
    summarize_by_fold(
        support_counts,
        ["dataset", "slice"],
        ["share_of_fold", "item_support_median"],
    ).to_csv(output / "support_slice_counts_summary_ci.csv", index=False)

    stability_rows: list[dict[str, object]] = []
    target = high_by_fold[high_by_fold["model"].isin(["CTW+NN", "AKT", "simpleKT", "stableKT"])]
    summary_for_stability = summarize_by_fold(
        target,
        ["dataset", "prior_strength", "rasch_prior_precision", "model"],
        ["delta_acc_vs_anchor", "delta_brier_vs_anchor", "delta_logloss_vs_anchor", "delta_ece_vs_anchor"],
    )
    for keys, group in summary_for_stability.groupby(["dataset", "model"], observed=False):
        row = {"dataset": keys[0], "model": keys[1], "n_settings": int(len(group))}
        for metric in ("delta_acc_vs_anchor", "delta_brier_vs_anchor", "delta_logloss_vs_anchor", "delta_ece_vs_anchor"):
            column = f"{metric}_mean"
            row[f"{metric}_min_mean"] = float(group[column].min())
            row[f"{metric}_max_mean"] = float(group[column].max())
            if metric == "delta_acc_vs_anchor":
                row[f"{metric}_all_favorable"] = bool((group[column] > 0).all())
            else:
                row[f"{metric}_all_favorable"] = bool((group[column] < 0).all())
        stability_rows.append(row)
    pd.DataFrame(stability_rows).to_csv(output / "prior_sensitivity_stability_summary.csv", index=False)

    write_prior_transfer_summary(high_by_fold, global_by_fold, output)
    write_existing_artifact_index(repo, output)
    config = {
        "repo": str(repo),
        "common_dir": str(common_dir),
        "output": str(output),
        "prior_strengths": prior_strengths,
        "rasch_precisions": rasch_precisions,
        "ece_bins": args.ece_bins,
        "common_sample_key": ["dataset", "fold", "orirow", "qidx"],
    }
    (output / "analysis_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_readme(output, config)
    print(f"Wrote anchor sensitivity/OOD outputs to {output}")


if __name__ == "__main__":
    main()

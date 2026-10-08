#!/usr/bin/env python3
"""Build one rebuttal evaluation cohort and run anchor robustness analyses.

The script uses existing model predictions and training sequences. It does not
retrain a neural KT model. Two independent item-aware anchors are constructed:

* ItemMean: an empirical-Bayes item correctness estimate from training folds.
* ItemRasch: ItemMean logits plus a causal online 1PL-style ability update.

Every metric, entropy bucket, and calibration result is computed on the same
intersection of CTW, CTW+NN, AKT, simpleKT, and stableKT predictions.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


DATASETS = ("nips_task34", "algebra2005")
CORE_MODELS = ("akt", "simplekt", "stablekt")
MODEL_COLUMNS = {
    "CTW": "p_ctw",
    "CTW+NN": "p_ctw_nn",
    "ItemMean": "p_item_mean",
    "ItemRasch": "p_item_rasch",
    "AKT": "p_akt",
    "simpleKT": "p_simplekt",
    "stableKT": "p_stablekt",
}
AXES = {
    "ctw": ("p_ctw", "CTW"),
    "item_mean": ("p_item_mean", "ItemMean"),
    "item_rasch": ("p_item_rasch", "ItemRasch"),
}
TRAIN_FILES = {
    "nips_task34": "data/nips_task34/train_data/train_valid_sequences_quelevel.csv",
    "algebra2005": "data/algebra2005/train_valid_sequences_quelevel.csv",
}
PAPER_REPORTED_TOTALS = {
    "nips_task34": 1_354_735,
    "algebra2005": 464_725,
}
BAND_LABELS = ("0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0")
QUINTILE_LABELS = ("Q1/5", "Q2/5", "Q3/5", "Q4/5", "Q5/5")
BAND_EDGES = np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0000000001])
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("runs/kt_ctw_question_alignment/repairs_20260724/manifest_with_algebra_akt_fold1_repair.csv"),
    )
    parser.add_argument("--prior-strength", type=float, default=20.0)
    parser.add_argument("--rasch-prior-precision", type=float, default=1.0)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--output", type=Path, default=Path("runs/rebuttal_unified_item_anchor_20260725"))
    parser.add_argument("--check-inputs", action="store_true", help="Report missing inputs and exit.")
    return parser.parse_args()


def check_inputs(repo: Path, manifest_path: Path) -> int:
    required = [manifest_path] + [repo / value for value in TRAIN_FILES.values()]
    required += [repo / f"runs/window_exports_20260427/strict_bands/{dataset}/strict_anchor_question_level.csv" for dataset in DATASETS]
    missing = [path for path in required if not path.exists()]
    if missing:
        print("Missing required inputs (see README.md 'Expected input layout'; override with --repo/--manifest):")
        for path in missing:
            print(f"- {path}")
    else:
        print("All default required inputs are present.")
    return 0


def normalize_item(value: object) -> str:
    text = str(value).strip()
    try:
        return str(int(float(text)))
    except (TypeError, ValueError):
        return text


def tokens(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return str(value).split(",")


def safe_int(value: str, default: int = -1) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def logit(p: np.ndarray | float) -> np.ndarray | float:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def binary_entropy_bits(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def fixed_bands(entropy: np.ndarray) -> pd.Categorical:
    codes = np.digitize(entropy, BAND_EDGES[1:-1], right=False)
    return pd.Categorical.from_codes(codes, categories=BAND_LABELS, ordered=True)


def equal_frequency_bands(entropy: pd.Series) -> pd.Categorical:
    # Ranking first guarantees equal support even when an item-only anchor has ties.
    rank = entropy.rank(method="first")
    return pd.qcut(rank, q=5, labels=QUINTILE_LABELS)


def load_item_counts(path: Path) -> tuple[list[dict[str, list[int]]], dict[str, list[int]], dict[str, int]]:
    frame = pd.read_csv(path, usecols=["fold", "questions", "responses", "selectmasks"])
    by_fold: list[dict[str, list[int]]] = [defaultdict(lambda: [0, 0]) for _ in range(5)]
    global_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    interaction_counts = defaultdict(int)

    for row in frame.itertuples(index=False):
        fold = int(row.fold)
        questions = tokens(row.questions)
        responses = tokens(row.responses)
        masks = tokens(row.selectmasks)
        for question, response, mask in zip(questions, responses, masks):
            item = normalize_item(question)
            y = safe_int(response)
            selected = safe_int(mask)
            if item == "-1" or y not in (0, 1) or selected < 0:
                continue
            by_fold[fold][item][0] += 1
            by_fold[fold][item][1] += y
            global_counts[item][0] += 1
            global_counts[item][1] += y
            interaction_counts[fold] += 1
    return by_fold, global_counts, dict(interaction_counts)


def item_parameters(
    held_out_fold: int,
    by_fold: list[dict[str, list[int]]],
    global_counts: dict[str, list[int]],
    prior_strength: float,
) -> tuple[dict[str, float], float, dict[str, int]]:
    held_out = by_fold[held_out_fold]
    train_n = sum(v[0] for v in global_counts.values()) - sum(v[0] for v in held_out.values())
    train_y = sum(v[1] for v in global_counts.values()) - sum(v[1] for v in held_out.values())
    global_rate = train_y / train_n
    probabilities: dict[str, float] = {}
    supports: dict[str, int] = {}
    for item, (all_n, all_y) in global_counts.items():
        held_n, held_y = held_out.get(item, (0, 0))
        n = all_n - held_n
        y = all_y - held_y
        probabilities[item] = (y + prior_strength * global_rate) / (n + prior_strength)
        supports[item] = n
    return probabilities, float(global_rate), supports


def online_rasch_predictions(
    frame: pd.DataFrame,
    prior_precision: float,
) -> np.ndarray:
    result = np.empty(len(frame), dtype=float)
    positions = np.empty(len(frame), dtype=np.int32)
    for _, indices in frame.groupby("orirow", sort=False).groups.items():
        ordered = np.asarray(list(indices), dtype=np.int64)
        theta = 0.0
        precision = float(prior_precision)
        for position, idx in enumerate(ordered, start=1):
            base = float(logit(float(frame.at[idx, "p_item_mean"])))
            p = float(sigmoid(base + theta))
            result[idx] = p
            positions[idx] = position
            y = int(frame.at[idx, "y_true"])
            precision += max(p * (1.0 - p), 1e-6)
            theta = float(np.clip(theta + (y - p) / precision, -4.0, 4.0))
    frame["seq_pos"] = positions
    return result


def resolve_shard(manifest: pd.DataFrame, dataset: str, model: str, fold: int) -> Path:
    rows = manifest[
        (manifest["dataset"] == dataset)
        & (manifest["model"] == model)
        & (manifest["emb_type"] == "qid")
        & (manifest["fold"] == fold)
        & (manifest["split"] == "question_window")
        & (manifest["status"] == "ok")
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one shard for {dataset}/{model}/fold{fold}, found {len(rows)}")
    return Path(rows.iloc[0]["output_shard"])


def build_common_fold(
    repo: Path,
    manifest: pd.DataFrame,
    dataset: str,
    fold: int,
    item_probs: dict[str, float],
    global_rate: float,
    item_support: dict[str, int],
    prior_precision: float,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    strict_path = repo / f"runs/window_exports_20260427/strict_bands/{dataset}/strict_anchor_question_level.csv"
    if not strict_path.exists():
        raise FileNotFoundError(f"Required strict anchor input missing: {strict_path}. Override --repo; see README.md 'Expected input layout'.")
    strict = pd.read_csv(strict_path)
    strict = strict[strict["fold"] == fold].copy()
    strict = strict.rename(columns={"p_fused": "p_ctw_nn"})
    strict = strict[["fold", "orirow", "qidx", "y_true", "p_ctw", "p_ctw_nn"]]
    if strict.duplicated(["orirow", "qidx"]).any():
        raise ValueError(f"Duplicate strict keys in {dataset}/fold{fold}")

    coverage = [
        {
            "dataset": dataset,
            "fold": fold,
            "scope": "strict_ctw_ctwnn",
            "n": len(strict),
        }
    ]
    common = strict
    reference_items: pd.Series | None = None
    for model in CORE_MODELS:
        shard = resolve_shard(manifest, dataset, model, fold)
        if not shard.is_absolute():
            shard = repo / shard
        if not shard.exists():
            raise FileNotFoundError(f"Required aligned shard missing: {shard}. Override --repo/--manifest; see README.md 'Expected input layout'.")
        columns = ["orirow", "qidx", "kt_questions", "ctw_y_true", "kt_late_mean"]
        model_frame = pd.read_parquet(shard, columns=columns).dropna()
        model_frame = model_frame.drop_duplicates(["orirow", "qidx"], keep="first")
        coverage.append(
            {
                "dataset": dataset,
                "fold": fold,
                "scope": f"{model}_shard",
                "n": len(model_frame),
            }
        )
        model_frame["item_id"] = model_frame["kt_questions"].map(normalize_item)
        model_frame = model_frame.rename(
            columns={"ctw_y_true": f"y_{model}", "kt_late_mean": f"p_{model}"}
        )
        keep = ["orirow", "qidx", "item_id", f"y_{model}", f"p_{model}"]
        common = common.merge(model_frame[keep], on=["orirow", "qidx"], how="inner", validate="one_to_one")
        if not (common["y_true"].astype(int) == common[f"y_{model}"].astype(int)).all():
            raise ValueError(f"Label mismatch for {dataset}/{model}/fold{fold}")
        if reference_items is None:
            reference_items = common["item_id"].copy()
            common = common.rename(columns={"item_id": "item_id_ref"})
        else:
            if not (common["item_id_ref"] == common["item_id"]).all():
                raise ValueError(f"Item mismatch for {dataset}/{model}/fold{fold}")
            common = common.drop(columns=["item_id"])
        common = common.drop(columns=[f"y_{model}"])

    common = common.rename(columns={"item_id_ref": "item_id"})
    common = common.sort_values(["orirow", "qidx"]).reset_index(drop=True)
    common["dataset"] = dataset
    common["p_item_mean"] = common["item_id"].map(item_probs).fillna(global_rate).astype(float)
    common["item_train_support"] = common["item_id"].map(item_support).fillna(0).astype(int)
    common["p_item_rasch"] = online_rasch_predictions(common, prior_precision)
    coverage.append(
        {
            "dataset": dataset,
            "fold": fold,
            "scope": "unified_common_intersection",
            "n": len(common),
        }
    )
    return common, coverage


def basic_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    y = np.asarray(y, dtype=float)
    return {
        "acc": float(((p >= 0.5) == y.astype(bool)).mean()),
        "brier": float(np.square(p - y).mean()),
        "logloss": float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean()),
        "mean_prob": float(p.mean()),
        "observed_rate": float(y.mean()),
    }


def calibration_metrics(
    y: np.ndarray,
    p: np.ndarray,
    n_bins: int,
) -> tuple[dict[str, float], list[dict[str, float | int]]]:
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    y = np.asarray(y, dtype=float)
    codes = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    mce = 0.0
    bins: list[dict[str, float | int]] = []
    for code in range(n_bins):
        mask = codes == code
        n = int(mask.sum())
        if n == 0:
            continue
        confidence = float(p[mask].mean())
        accuracy = float(y[mask].mean())
        gap = confidence - accuracy
        ece += (n / len(y)) * abs(gap)
        mce = max(mce, abs(gap))
        bins.append(
            {
                "calibration_bin": code,
                "calibration_lower": code / n_bins,
                "calibration_upper": (code + 1) / n_bins,
                "n": n,
                "mean_prob": confidence,
                "observed_rate": accuracy,
                "signed_gap": gap,
                "absolute_gap": abs(gap),
            }
        )
    return {
        "ece": float(ece),
        "mce": float(mce),
        "signed_calibration_gap": float(p.mean() - y.mean()),
    }, bins


def evaluate_common_fold(
    frame: pd.DataFrame,
    ece_bins: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    metric_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    count_rows: list[dict[str, object]] = []
    dataset = str(frame["dataset"].iat[0])
    fold = int(frame["fold"].iat[0])

    for axis, (axis_column, reference_model) in AXES.items():
        entropy = pd.Series(binary_entropy_bits(frame[axis_column].to_numpy()), index=frame.index)
        schemes = {
            "fixed_width": fixed_bands(entropy.to_numpy()),
            "equal_frequency": equal_frequency_bands(entropy),
        }
        for scheme, bands in schemes.items():
            local = frame.assign(entropy_bits=entropy, band=bands)
            for band, band_frame in local.groupby("band", observed=False):
                if band_frame.empty:
                    continue
                count_rows.append(
                    {
                        "dataset": dataset,
                        "fold": fold,
                        "axis": axis,
                        "scheme": scheme,
                        "band": str(band),
                        "n": len(band_frame),
                        "entropy_mean_bits": float(band_frame["entropy_bits"].mean()),
                        "entropy_min_bits": float(band_frame["entropy_bits"].min()),
                        "entropy_max_bits": float(band_frame["entropy_bits"].max()),
                    }
                )
                y = band_frame["y_true"].to_numpy(dtype=int)
                per_model: dict[str, dict[str, float]] = {}
                per_model_bins: dict[str, list[dict[str, float | int]]] = {}
                for model, probability_column in MODEL_COLUMNS.items():
                    p = band_frame[probability_column].to_numpy(dtype=float)
                    values = basic_metrics(y, p)
                    calibration, bins = calibration_metrics(y, p, ece_bins)
                    values.update(calibration)
                    per_model[model] = values
                    per_model_bins[model] = bins

                reference = per_model[reference_model]
                for model, values in per_model.items():
                    row = {
                        "dataset": dataset,
                        "fold": fold,
                        "axis": axis,
                        "reference_model": reference_model,
                        "scheme": scheme,
                        "band": str(band),
                        "model": model,
                        "n": len(band_frame),
                        "entropy_mean_bits": float(band_frame["entropy_bits"].mean()),
                        **values,
                    }
                    for metric in ("acc", "brier", "logloss", "ece", "mce", "signed_calibration_gap"):
                        row[f"delta_{metric}_vs_anchor"] = values[metric] - reference[metric]
                    metric_rows.append(row)
                    for calibration_bin in per_model_bins[model]:
                        calibration_rows.append(
                            {
                                "dataset": dataset,
                                "fold": fold,
                                "axis": axis,
                                "reference_model": reference_model,
                                "scheme": scheme,
                                "band": str(band),
                                "model": model,
                                **calibration_bin,
                            }
                        )
    return metric_rows, calibration_rows, count_rows


def evaluate_global_fold(frame: pd.DataFrame, ece_bins: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    y = frame["y_true"].to_numpy(dtype=int)
    for model, probability_column in MODEL_COLUMNS.items():
        values = basic_metrics(y, frame[probability_column].to_numpy(dtype=float))
        calibration, _ = calibration_metrics(y, frame[probability_column].to_numpy(dtype=float), ece_bins)
        rows.append(
            {
                "dataset": str(frame["dataset"].iat[0]),
                "fold": int(frame["fold"].iat[0]),
                "model": model,
                "n": len(frame),
                **values,
                **calibration,
            }
        )
    return rows


def summarize_global_folds(metrics: pd.DataFrame) -> pd.DataFrame:
    value_columns = ["acc", "brier", "logloss", "ece", "mce", "signed_calibration_gap"]
    rows: list[dict[str, object]] = []
    for keys, group in metrics.groupby(["dataset", "model"]):
        row = {"dataset": keys[0], "model": keys[1], "n_total": int(group["n"].sum()), "n_folds": len(group)}
        for column in value_columns:
            values = group[column].astype(float)
            mean = float(values.mean())
            std = float(values.std(ddof=1))
            se = std / math.sqrt(len(values))
            halfwidth = float(stats.t.ppf(0.975, len(values) - 1) * se)
            row[f"{column}_mean"] = mean
            row[f"{column}_std"] = std
            row[f"{column}_se"] = se
            row[f"{column}_ci95_halfwidth"] = halfwidth
            row[f"{column}_ci95_low"] = mean - halfwidth
            row[f"{column}_ci95_high"] = mean + halfwidth
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_folds(metrics: pd.DataFrame) -> pd.DataFrame:
    value_columns = [
        "acc",
        "brier",
        "logloss",
        "ece",
        "mce",
        "signed_calibration_gap",
        "delta_acc_vs_anchor",
        "delta_brier_vs_anchor",
        "delta_logloss_vs_anchor",
        "delta_ece_vs_anchor",
        "delta_mce_vs_anchor",
        "delta_signed_calibration_gap_vs_anchor",
    ]
    group_columns = ["dataset", "axis", "reference_model", "scheme", "band", "model"]
    output: list[dict[str, object]] = []
    for keys, group in metrics.groupby(group_columns, observed=False):
        row = dict(zip(group_columns, keys))
        row["n_total"] = int(group["n"].sum())
        row["n_folds"] = int(group["fold"].nunique())
        row["entropy_mean_bits"] = float(np.average(group["entropy_mean_bits"], weights=group["n"]))
        for column in value_columns:
            values = group[column].astype(float)
            mean = float(values.mean())
            std = float(values.std(ddof=1))
            se = std / math.sqrt(len(values))
            halfwidth = float(stats.t.ppf(0.975, len(values) - 1) * se)
            row[f"{column}_mean"] = mean
            row[f"{column}_std"] = std
            row[f"{column}_se"] = se
            row[f"{column}_ci95_halfwidth"] = halfwidth
            row[f"{column}_ci95_low"] = mean - halfwidth
            row[f"{column}_ci95_high"] = mean + halfwidth
        output.append(row)
    return pd.DataFrame(output)


def high_low_tests(metrics: pd.DataFrame) -> pd.DataFrame:
    tests: list[dict[str, object]] = []
    tested_metrics = (
        "delta_acc_vs_anchor",
        "delta_brier_vs_anchor",
        "delta_logloss_vs_anchor",
        "delta_ece_vs_anchor",
    )
    groups = ["dataset", "axis", "reference_model", "model"]
    fixed = metrics[metrics["scheme"] == "fixed_width"]
    for keys, group in fixed.groupby(groups, observed=False):
        for metric in tested_metrics:
            pivot = group.pivot(index="fold", columns="band", values=metric)
            if "0.0-0.2" not in pivot or "0.8-1.0" not in pivot:
                continue
            pair = pivot[["0.0-0.2", "0.8-1.0"]].dropna()
            if len(pair) < 2:
                continue
            t_stat, p_value = stats.ttest_rel(pair["0.8-1.0"], pair["0.0-0.2"])
            tests.append(
                {
                    **dict(zip(groups, keys)),
                    "metric": metric,
                    "n_pairs": len(pair),
                    "low_mean": float(pair["0.0-0.2"].mean()),
                    "high_mean": float(pair["0.8-1.0"].mean()),
                    "high_minus_low_mean": float((pair["0.8-1.0"] - pair["0.0-0.2"]).mean()),
                    "t_stat": float(t_stat),
                    "p_value_two_sided": float(p_value),
                }
            )
    return pd.DataFrame(tests)


def sequence_summary(common_paths: Iterable[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in common_paths:
        frame = pd.read_parquet(path, columns=["dataset", "fold", "seq_pos", "p_ctw", "p_item_mean", "p_item_rasch"])
        dataset = str(frame["dataset"].iat[0])
        fold = int(frame["fold"].iat[0])
        for axis, (probability_column, _) in AXES.items():
            entropy = binary_entropy_bits(frame[probability_column].to_numpy())
            high = frame[np.asarray(fixed_bands(entropy) == "0.8-1.0")]
            if high.empty:
                continue
            positions = high["seq_pos"].astype(float)
            rows.append(
                {
                    "dataset": dataset,
                    "fold": fold,
                    "axis": axis,
                    "n": len(high),
                    "share_of_common_sample": len(high) / len(frame),
                    "seq_pos_mean": float(positions.mean()),
                    "seq_pos_median": float(positions.median()),
                    "seq_pos_p90": float(positions.quantile(0.90)),
                    "seq_pos_le_20_share": float((positions <= 20).mean()),
                    "seq_pos_gt_20_share": float((positions > 20).mean()),
                }
            )
    per_fold = pd.DataFrame(rows)
    totals: list[dict[str, object]] = []
    for keys, group in per_fold.groupby(["dataset", "axis"]):
        weights = group["n"].to_numpy()
        totals.append(
            {
                "dataset": keys[0],
                "fold": "all",
                "axis": keys[1],
                "n": int(group["n"].sum()),
                "share_of_common_sample": float(np.average(group["share_of_common_sample"], weights=weights)),
                "seq_pos_mean": float(np.average(group["seq_pos_mean"], weights=weights)),
                "seq_pos_median": float(np.average(group["seq_pos_median"], weights=weights)),
                "seq_pos_p90": float(np.average(group["seq_pos_p90"], weights=weights)),
                "seq_pos_le_20_share": float(np.average(group["seq_pos_le_20_share"], weights=weights)),
                "seq_pos_gt_20_share": float(np.average(group["seq_pos_gt_20_share"], weights=weights)),
            }
        )
    return pd.concat([per_fold, pd.DataFrame(totals)], ignore_index=True)


def write_readme(
    output: Path,
    manifest_path: Path,
    prior_strength: float,
    prior_precision: float,
    ece_bins: int,
) -> None:
    content = f"""# Unified Rebuttal Analysis

All outputs in this directory use one exact sample contract. No neural KT
model was retrained.

## Common sample

The key is `(dataset, fold, orirow, qidx)`. Each row must be present in CTW,
CTW+NN, AKT, simpleKT, and stableKT exports and have identical label and item
identity. Independent item anchors are then added to this intersection.

Sequence position is reconstructed within each `orirow` after sorting by
`qidx`; it is not the raw `qidx` value, which is an export-side alignment key.

The submitted paper reports counts from an earlier export. The
`submitted_paper_reported_total` rows in `sample_scope_audit.csv` preserve
those numbers for provenance; all metrics in this directory use only the
`unified_common_intersection` rows from the repaired manifest below.

Manifest: `{manifest_path}`

## Independent anchors

- `ItemMean`: empirical-Bayes item correctness from the four training folds
  other than the current validation fold. The prior is centered at the
  training-fold global correctness rate with strength `{prior_strength:g}`.
- `ItemRasch`: a causal online 1PL-style predictor. It starts from the
  ItemMean logit and updates one scalar sequence ability after each observed
  response using a Laplace/Elo-style update. Prior precision is
  `{prior_precision:g}`. The current response is never used before prediction.

Neither anchor uses CTW or a neural model prediction.

## Calibration

Within-band ECE uses `{ece_bins}` fixed probability bins over `[0, 1]` inside
each entropy band. `calibration_reliability_bins.csv` stores every component
needed to reproduce ECE. ECE is reported with Brier and Log-loss because ECE
alone is not a proper score.

## Files

- `common_samples/*.parquet`: exact shared evaluation rows and probabilities.
- `sample_scope_audit.csv`: source coverage and final intersection counts.
- `item_anchor_training_summary.csv`: item-anchor training coverage.
- `band_counts_by_fold.csv`: fixed-width and equal-frequency support.
- `global_metrics_by_fold.csv`: global strength of all anchors and core models.
- `global_metrics_fold_summary_ci.csv`: five-fold global summaries.
- `band_metrics_by_fold.csv`: ACC, Brier, Log-loss, and within-band ECE.
- `band_metrics_fold_summary_ci.csv`: five-fold mean, SE, and 95% t intervals.
- `calibration_reliability_bins.csv`: within-band reliability bins.
- `fixed_band_high_vs_low_paired_tests.csv`: paired fold tests.
- `high_entropy_sequence_position_unified.csv`: position analysis on the same cohort.
"""
    (output / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else repo / args.manifest
    if args.check_inputs:
        raise SystemExit(check_inputs(repo, manifest_path))
    if not manifest_path.exists():
        raise FileNotFoundError(f"Required manifest missing: {manifest_path}. Override with --manifest/--repo; see README.md 'Expected input layout'.")
    output = args.output if args.output.is_absolute() else repo / args.output
    common_dir = output / "common_samples"
    common_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path)

    all_metrics: list[dict[str, object]] = []
    all_global_metrics: list[dict[str, object]] = []
    all_calibration: list[dict[str, object]] = []
    all_band_counts: list[dict[str, object]] = []
    all_coverage: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    common_paths: list[Path] = []

    for dataset in DATASETS:
        by_fold, global_counts, interaction_counts = load_item_counts(repo / TRAIN_FILES[dataset])
        for fold in range(5):
            item_probs, global_rate, supports = item_parameters(
                fold, by_fold, global_counts, args.prior_strength
            )
            training_rows.append(
                {
                    "dataset": dataset,
                    "fold": fold,
                    "training_interactions": sum(interaction_counts.values()) - interaction_counts[fold],
                    "training_items": sum(1 for value in supports.values() if value > 0),
                    "global_correctness": global_rate,
                    "prior_strength": args.prior_strength,
                    "rasch_prior_precision": args.rasch_prior_precision,
                }
            )
            common, coverage = build_common_fold(
                repo,
                manifest,
                dataset,
                fold,
                item_probs,
                global_rate,
                supports,
                args.rasch_prior_precision,
            )
            all_coverage.extend(coverage)
            common_path = common_dir / f"{dataset}_fold{fold}.parquet"
            common.to_parquet(common_path, index=False)
            common_paths.append(common_path)
            all_global_metrics.extend(evaluate_global_fold(common, args.ece_bins))
            metrics, calibration, band_counts = evaluate_common_fold(common, args.ece_bins)
            all_metrics.extend(metrics)
            all_calibration.extend(calibration)
            all_band_counts.extend(band_counts)

    coverage = pd.DataFrame(all_coverage)
    for dataset, total in PAPER_REPORTED_TOTALS.items():
        coverage.loc[len(coverage)] = {
            "dataset": dataset,
            "fold": -1,
            "scope": "submitted_paper_reported_total",
            "n": total,
        }
    coverage.to_csv(output / "sample_scope_audit.csv", index=False)
    pd.DataFrame(training_rows).to_csv(output / "item_anchor_training_summary.csv", index=False)

    global_metrics = pd.DataFrame(all_global_metrics)
    global_metrics.to_csv(output / "global_metrics_by_fold.csv", index=False)
    summarize_global_folds(global_metrics).to_csv(output / "global_metrics_fold_summary_ci.csv", index=False)

    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(output / "band_metrics_by_fold.csv", index=False)
    summary = summarize_folds(metrics)
    summary.to_csv(output / "band_metrics_fold_summary_ci.csv", index=False)
    pd.DataFrame(all_calibration).to_csv(output / "calibration_reliability_bins.csv", index=False)
    pd.DataFrame(all_band_counts).to_csv(output / "band_counts_by_fold.csv", index=False)
    high_low_tests(metrics).to_csv(output / "fixed_band_high_vs_low_paired_tests.csv", index=False)
    sequence_summary(common_paths).to_csv(output / "high_entropy_sequence_position_unified.csv", index=False)

    metadata = {
        "manifest": str(manifest_path),
        "datasets": list(DATASETS),
        "core_models": list(CORE_MODELS),
        "model_columns": MODEL_COLUMNS,
        "axes": AXES,
        "prior_strength": args.prior_strength,
        "rasch_prior_precision": args.rasch_prior_precision,
        "ece_bins": args.ece_bins,
        "common_sample_key": ["dataset", "fold", "orirow", "qidx"],
        "sequence_position": "one-based rank of qidx within orirow after common-sample intersection",
        "coverage_scope": "unified_common_intersection",
    }
    (output / "analysis_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_readme(
        output,
        manifest_path,
        args.prior_strength,
        args.rasch_prior_precision,
        args.ece_bins,
    )
    print(f"Wrote unified rebuttal outputs to {output}")


if __name__ == "__main__":
    main()

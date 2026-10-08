#!/usr/bin/env python3
"""Postprocess existing local artifacts for rebuttal tables and figures.

This script does not train models. It reads already aligned prediction shards
and exported uncertainty files, then writes aggregate CSV/PNG artifacts.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover - optional dependency fallback
    stats = None


BAND_EDGES = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0000000001])
BAND_LABELS = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
EPS = 1e-15


def binary_entropy_bits(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(float), EPS, 1.0 - EPS)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def fixed_entropy_band(entropy_bits: np.ndarray) -> pd.Categorical:
    idx = np.digitize(entropy_bits, BAND_EDGES[1:-1], right=False)
    return pd.Categorical.from_codes(idx, categories=BAND_LABELS, ordered=True)


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p.astype(float), EPS, 1.0 - EPS)
    y = y.astype(float)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def metric_row(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p.astype(float), EPS, 1.0 - EPS)
    pred = p >= 0.5
    return {
        "acc": float((pred == y.astype(bool)).mean()),
        "brier": float(np.square(p - y.astype(float)).mean()),
        "logloss": logloss(y, p),
        "mean_prob": float(p.mean()),
    }


def resolve_manifest(repo: Path, manifest_arg: str | None) -> Path:
    if manifest_arg:
        path = repo / manifest_arg
        if not path.exists():
            raise FileNotFoundError(f"Required aligned shard missing: {path}. Override --repo/--manifest; see README.md 'Expected input layout'.")
        return path
    repaired = repo / "runs/kt_ctw_question_alignment/repairs_20260724/manifest_with_algebra_akt_fold1_repair.csv"
    if repaired.exists():
        return repaired
    return repo / "runs/kt_ctw_question_alignment/manifest.csv"


def read_manifest(repo: Path, manifest_path: Path, datasets: set[str]) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    rows = manifest[
        (manifest["split"] == "question_window")
        & (manifest["status"] == "ok")
        & (manifest["dataset"].isin(datasets))
    ].copy()
    if rows.empty:
        raise ValueError("No matching question_window rows found in manifest.")
    rows["output_shard"] = rows["output_shard"].map(lambda p: str(Path(p)))
    return rows


def iter_model_shards(rows: pd.DataFrame, repo: Path) -> Iterable[tuple[pd.Series, Path]]:
    for _, row in rows.iterrows():
        path = Path(row["output_shard"])
        if not path.is_absolute():
            path = repo / path
        if not path.exists():
            raise FileNotFoundError(path)
        yield row, path


def add_fused_strict_anchor_records(repo: Path, datasets: Iterable[str], records: list[dict[str, object]]) -> None:
    for dataset in datasets:
        path = repo / f"runs/window_exports_20260427/strict_bands/{dataset}/strict_anchor_question_level.csv"
        if not path.exists():
            continue
        df = pd.read_csv(
            path,
            usecols=["fold", "y_true", "p_ctw", "p_fused", "ctw_entropy_band_label"],
        ).dropna(subset=["y_true", "p_ctw", "p_fused", "ctw_entropy_band_label"])
        for (fold, band_label), group in df.groupby(["fold", "ctw_entropy_band_label"], observed=False):
            gy = group["y_true"].astype(int).to_numpy()
            kt = metric_row(gy, group["p_fused"].astype(float).to_numpy())
            ctw = metric_row(gy, group["p_ctw"].astype(float).to_numpy())
            rec = {
                "dataset": dataset,
                "model": "ctw+nn",
                "emb_type": "fused",
                "fold": int(fold),
                "ctw_entropy_band_label": str(band_label),
                "n": int(len(group)),
                "kt_acc": kt["acc"],
                "kt_brier": kt["brier"],
                "kt_logloss": kt["logloss"],
                "kt_mean_prob": kt["mean_prob"],
                "ctw_acc": ctw["acc"],
                "ctw_brier": ctw["brier"],
                "ctw_logloss": ctw["logloss"],
                "ctw_mean_prob": ctw["mean_prob"],
            }
            rec["delta_acc_vs_ctw"] = rec["kt_acc"] - rec["ctw_acc"]
            rec["delta_brier_vs_ctw"] = rec["kt_brier"] - rec["ctw_brier"]
            rec["delta_logloss_vs_ctw"] = rec["kt_logloss"] - rec["ctw_logloss"]
            records.append(rec)


def compute_fixed_band_metrics(repo: Path, out_dir: Path, rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_fold_records: list[dict[str, object]] = []
    columns = ["ctw_y_true", "kt_late_mean", "ctw_late_mean", "fold"]

    for row, path in iter_model_shards(rows, repo):
        df = pd.read_parquet(path, columns=columns)
        df = df.dropna(subset=["ctw_y_true", "kt_late_mean", "ctw_late_mean"])
        if df.empty:
            continue
        y = df["ctw_y_true"].astype(int).to_numpy()
        p_kt = df["kt_late_mean"].astype(float).to_numpy()
        p_ctw = df["ctw_late_mean"].astype(float).to_numpy()
        entropy = binary_entropy_bits(p_ctw)
        df = pd.DataFrame(
            {
                "y": y,
                "p_kt": p_kt,
                "p_ctw": p_ctw,
                "entropy_band": fixed_entropy_band(entropy),
            }
        )

        for band_label, group in df.groupby("entropy_band", observed=False):
            if group.empty:
                continue
            gy = group["y"].to_numpy()
            kt = metric_row(gy, group["p_kt"].to_numpy())
            ctw = metric_row(gy, group["p_ctw"].to_numpy())
            rec = {
                "dataset": row["dataset"],
                "model": row["model"],
                "emb_type": row["emb_type"],
                "fold": int(row["fold"]),
                "ctw_entropy_band_label": str(band_label),
                "n": int(len(group)),
                "kt_acc": kt["acc"],
                "kt_brier": kt["brier"],
                "kt_logloss": kt["logloss"],
                "kt_mean_prob": kt["mean_prob"],
                "ctw_acc": ctw["acc"],
                "ctw_brier": ctw["brier"],
                "ctw_logloss": ctw["logloss"],
                "ctw_mean_prob": ctw["mean_prob"],
            }
            rec["delta_acc_vs_ctw"] = rec["kt_acc"] - rec["ctw_acc"]
            rec["delta_brier_vs_ctw"] = rec["kt_brier"] - rec["ctw_brier"]
            rec["delta_logloss_vs_ctw"] = rec["kt_logloss"] - rec["ctw_logloss"]
            per_fold_records.append(rec)

    add_fused_strict_anchor_records(repo, sorted(rows["dataset"].unique()), per_fold_records)

    per_fold = pd.DataFrame(per_fold_records)
    per_fold.to_csv(out_dir / "fixed_entropy_band_metrics_by_fold.csv", index=False)

    metric_cols = [
        "kt_acc",
        "kt_brier",
        "kt_logloss",
        "ctw_acc",
        "ctw_brier",
        "ctw_logloss",
        "delta_acc_vs_ctw",
        "delta_brier_vs_ctw",
        "delta_logloss_vs_ctw",
    ]
    grouped = per_fold.groupby(["dataset", "model", "emb_type", "ctw_entropy_band_label"], observed=False)
    rows_out: list[dict[str, object]] = []
    for keys, group in grouped:
        rec = {
            "dataset": keys[0],
            "model": keys[1],
            "emb_type": keys[2],
            "ctw_entropy_band_label": keys[3],
            "n_total": int(group["n"].sum()),
            "n_folds": int(group["fold"].nunique()),
        }
        for col in metric_cols:
            values = group[col].dropna().astype(float)
            n = len(values)
            mean = float(values.mean()) if n else np.nan
            std = float(values.std(ddof=1)) if n > 1 else np.nan
            se = float(std / math.sqrt(n)) if n > 1 else np.nan
            if n > 1 and stats is not None:
                tcrit = float(stats.t.ppf(0.975, df=n - 1))
            else:
                tcrit = 1.96
            rec[f"{col}_mean"] = mean
            rec[f"{col}_std"] = std
            rec[f"{col}_se"] = se
            rec[f"{col}_ci95_halfwidth"] = float(tcrit * se) if n > 1 else np.nan
        rows_out.append(rec)
    summary = pd.DataFrame(rows_out)
    summary.to_csv(out_dir / "fixed_entropy_band_metrics_mean_std_ci.csv", index=False)

    tests: list[dict[str, object]] = []
    low_label = "0.0-0.2"
    high_label = "0.8-1.0"
    for keys, group in per_fold.groupby(["dataset", "model", "emb_type"], observed=False):
        pivot = group.pivot(index="fold", columns="ctw_entropy_band_label")
        for metric in ["delta_acc_vs_ctw", "delta_brier_vs_ctw", "delta_logloss_vs_ctw"]:
            if (metric, low_label) not in pivot or (metric, high_label) not in pivot:
                continue
            paired = pd.DataFrame(
                {
                    "low": pivot[(metric, low_label)],
                    "high": pivot[(metric, high_label)],
                }
            ).dropna()
            if len(paired) < 2:
                continue
            diff = paired["high"] - paired["low"]
            if stats is not None:
                t_stat, p_value = stats.ttest_rel(paired["high"], paired["low"])
            else:
                t_stat, p_value = np.nan, np.nan
            tests.append(
                {
                    "dataset": keys[0],
                    "model": keys[1],
                    "emb_type": keys[2],
                    "metric": metric,
                    "low_band": low_label,
                    "high_band": high_label,
                    "n_pairs": int(len(paired)),
                    "low_mean": float(paired["low"].mean()),
                    "high_mean": float(paired["high"].mean()),
                    "high_minus_low_mean": float(diff.mean()),
                    "high_minus_low_std": float(diff.std(ddof=1)),
                    "t_stat": float(t_stat) if pd.notna(t_stat) else np.nan,
                    "p_value": float(p_value) if pd.notna(p_value) else np.nan,
                }
            )
    tests_df = pd.DataFrame(tests)
    tests_df.to_csv(out_dir / "fixed_entropy_band_high_vs_low_paired_tests.csv", index=False)
    return per_fold, summary, tests_df


def summarize_sequence_positions(repo: Path, out_dir: Path, datasets: Iterable[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    high_records: list[dict[str, object]] = []
    for dataset in datasets:
        path = repo / f"runs/window_exports_20260427/strict_bands/{dataset}/strict_anchor_question_level.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=["fold", "orirow", "qidx", "ctw_entropy_band_label"])
        df = df.sort_values(["fold", "orirow", "qidx"]).copy()
        df["seq_pos"] = df.groupby(["fold", "orirow"]).cumcount() + 1
        total = len(df)
        for band, group in df.groupby("ctw_entropy_band_label", observed=False):
            pos = group["seq_pos"].astype(float)
            rec = {
                "dataset": dataset,
                "ctw_entropy_band_label": band,
                "n": int(len(group)),
                "share_of_dataset": float(len(group) / total),
                "seq_pos_mean": float(pos.mean()),
                "seq_pos_median": float(pos.median()),
                "seq_pos_p75": float(pos.quantile(0.75)),
                "seq_pos_p90": float(pos.quantile(0.90)),
                "seq_pos_le_20_share": float((pos <= 20).mean()),
                "seq_pos_gt_20_share": float((pos > 20).mean()),
            }
            records.append(rec)
            if band == "0.8-1.0":
                high_records.append(rec.copy())

        for fold, group in df[df["ctw_entropy_band_label"] == "0.8-1.0"].groupby("fold"):
            pos = group["seq_pos"].astype(float)
            high_records.append(
                {
                    "dataset": dataset,
                    "fold": int(fold),
                    "ctw_entropy_band_label": "0.8-1.0",
                    "n": int(len(group)),
                    "share_of_dataset": float(len(group) / total),
                    "seq_pos_mean": float(pos.mean()),
                    "seq_pos_median": float(pos.median()),
                    "seq_pos_p75": float(pos.quantile(0.75)),
                    "seq_pos_p90": float(pos.quantile(0.90)),
                    "seq_pos_le_20_share": float((pos <= 20).mean()),
                    "seq_pos_gt_20_share": float((pos > 20).mean()),
                }
            )

    by_band = pd.DataFrame(records)
    high = pd.DataFrame(high_records)
    by_band.to_csv(out_dir / "sequence_position_by_entropy_band.csv", index=False)
    high.to_csv(out_dir / "high_entropy_sequence_position_summary.csv", index=False)
    return by_band, high


def plot_equal_frequency(repo: Path, out_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    src = repo / "runs/window_exports_20260427/refined_uncertainty_refresh/refined_uncertainty_bucket_delta_vs_ctw.csv"
    if not src.exists():
        return []
    df = pd.read_csv(src)
    keep_models = ["ctw+nn", "simplekt", "AKT", "stableKT", "ctw"]
    df = df[df["model_label"].isin(keep_models)].copy()
    df = df[df["uncertainty_source"].isin(["ctw", "refined"])]
    paths: list[Path] = []
    for dataset in sorted(df["dataset"].unique()):
        for source in sorted(df["uncertainty_source"].unique()):
            sub = df[(df["dataset"] == dataset) & (df["uncertainty_source"] == source)]
            if sub.empty:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
            for label, model_df in sub.groupby("model_label"):
                model_df = model_df.sort_values("bucket_index")
                axes[0].plot(model_df["bucket_index"] + 1, model_df["delta_acc_vs_ctw"], marker="o", label=label)
                axes[1].plot(model_df["bucket_index"] + 1, model_df["delta_brier_vs_ctw"], marker="o", label=label)
            axes[0].axhline(0, color="black", linewidth=0.8)
            axes[1].axhline(0, color="black", linewidth=0.8)
            axes[0].set_title("Delta ACC vs CTW")
            axes[1].set_title("Delta Brier vs CTW")
            for ax in axes:
                ax.set_xlabel("Equal-frequency bucket")
                ax.grid(True, alpha=0.25)
            axes[0].set_ylabel("Delta")
            axes[1].legend(loc="best", fontsize=8)
            fig.suptitle(f"{dataset} / {source} uncertainty")
            fig.tight_layout()
            out = out_dir / f"equal_frequency_delta_{dataset}_{source}.png"
            fig.savefig(out, dpi=200)
            plt.close(fig)
            paths.append(out)
    return paths


def write_readme(out_dir: Path, manifest: Path, rows: pd.DataFrame, plot_paths: list[Path]) -> None:
    lines = [
        "# Rebuttal Postprocess Outputs",
        "",
        "Generated from existing local artifacts only. No model training was run.",
        "",
        f"Manifest: `{manifest}`",
        f"Question-window shard rows in manifest: `{len(rows)}`",
        "",
        "## Output files",
        "",
        "- `fixed_entropy_band_metrics_by_fold.csv`: fold-level ACC/Brier/Log-loss and deltas vs CTW.",
        "- `fixed_entropy_band_metrics_mean_std_ci.csv`: mean/std/SE/95% CI across folds.",
        "- `fixed_entropy_band_high_vs_low_paired_tests.csv`: paired fold tests between `0.0-0.2` and `0.8-1.0` entropy bands.",
        "- `sequence_position_by_entropy_band.csv`: derived sequence-position distribution by fixed CTW entropy band.",
        "- `high_entropy_sequence_position_summary.csv`: high-entropy sequence-position summary overall and by fold.",
        "- `equal_frequency_delta_*.png`: equal-frequency bucket delta plots from existing refined uncertainty exports.",
        "",
        "## Notes",
        "",
        "- Fixed entropy bands are recomputed as binary entropy of `ctw_late_mean` from each aligned shard.",
        "- `ctw+nn` rows are appended from `strict_anchor_question_level.csv` using `p_fused` against `p_ctw`.",
        "- Deltas are computed on each model shard's matched rows: ACC uses `model - CTW`; Brier/Log-loss use `model - CTW`, so negative is better.",
        "- Sequence-position summaries use `strict_anchor_question_level.csv`; `seq_pos` is derived as `cumcount + 1` within each `(fold, orirow)` after sorting by `qidx`.",
        "",
        "## Plots",
        "",
    ]
    lines.extend(f"- `{p.name}`" for p in plot_paths)
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".", help="Repository root.")
    parser.add_argument("--manifest", default=None, help="Optional manifest CSV relative to repo or absolute.")
    parser.add_argument("--out-dir", default="runs/rebuttal_postprocess_existing_assets")
    parser.add_argument("--datasets", nargs="+", default=["nips_task34", "algebra2005"])
    parser.add_argument("--check-inputs", action="store_true", help="Report missing inputs and exit.")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    out_dir = (repo / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = set(args.datasets)
    if args.check_inputs:
        candidates = [repo / args.manifest] if args.manifest else [repo / "runs/kt_ctw_question_alignment/manifest.csv", repo / "runs/kt_ctw_question_alignment/repairs_20260724/manifest_with_algebra_akt_fold1_repair.csv"]
        candidates += [repo / f"runs/window_exports_20260427/strict_bands/{dataset}/strict_anchor_question_level.csv" for dataset in datasets]
        for path in candidates:
            if not path.exists():
                print(f"missing: {path}")
        return
    manifest_path = resolve_manifest(repo, args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Required manifest missing: {manifest_path}. Override with --manifest/--repo; see README.md 'Expected input layout'.")
    rows = read_manifest(repo, manifest_path, datasets)

    compute_fixed_band_metrics(repo, out_dir, rows)
    summarize_sequence_positions(repo, out_dir, args.datasets)
    plot_paths = plot_equal_frequency(repo, out_dir)
    write_readme(out_dir, manifest_path, rows, plot_paths)

    print(f"Wrote rebuttal postprocess outputs to {out_dir}")


if __name__ == "__main__":
    main()

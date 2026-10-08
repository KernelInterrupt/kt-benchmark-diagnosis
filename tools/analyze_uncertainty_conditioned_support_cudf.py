#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from pathlib import Path



DEFAULT_CTW_PATHS = {
    "nips_task34": {
        "question": "runs/ctw_benchmark_5fold/question_predictions.csv",
        "question_window": "runs/symbolic_hybrid_benchmark_parallel/strict_question_predictions/ctw/question_window_predictions.csv",
    },
    "algebra2005": {
        "question": "runs/ctw_benchmark_algebra2005_5fold/question_predictions.csv",
        "question_window": "runs/symbolic_hybrid_benchmark_parallel/strict_question_predictions_algebra2005/ctw/question_window_predictions.csv",
    },
    "assist2015": {
        "question": "runs/symbolic_hybrid_benchmark_parallel/strict_question_predictions_assist2015/ctw/question_predictions.csv",
        "question_window": "runs/symbolic_hybrid_benchmark_parallel/strict_question_predictions_assist2015/ctw/question_window_predictions.csv",
    },
}


def safe_prob(series, eps: float = 1e-7):
    series = series.astype("float64")
    series = series.where(series > eps, eps)
    series = series.where(series < 1.0 - eps, 1.0 - eps)
    return series


def binary_entropy_bits(prob):
    values = safe_prob(prob).to_cupy()
    entropy = -(values * cp.log2(values) + (1.0 - values) * cp.log2(1.0 - values))
    return cudf.Series(entropy)


def self_information_bits(prob, truth):
    prob = safe_prob(prob)
    truth = truth.astype("int8")
    true_prob = prob.where(truth == 1, 1.0 - prob)
    values = safe_prob(true_prob).to_cupy()
    return cudf.Series(-cp.log2(values))


def entropy_band(entropy):
    band = (entropy * 5.0).astype("int32")
    band = band.where(band < 5, 4)
    band = band.where(band >= 0, 0)
    return band


def band_label(index: int) -> str:
    return f"{index / 5:.1f}-{(index + 1) / 5:.1f}"


def resolve_ctw_path(repo: Path, dataset: str, split: str) -> Path:
    try:
        relpath = DEFAULT_CTW_PATHS[dataset][split]
    except KeyError as exc:
        raise KeyError(f"Missing default CTW path for dataset={dataset}, split={split}") from exc
    path = repo / relpath
    if not path.exists():
        raise FileNotFoundError(f"CTW anchor file not found for dataset={dataset}, split={split}: {path}")
    return path


def add_anchor_columns(frame):
    frame = frame.copy(deep=False)
    y = frame["y_true"].astype("float64")
    p = safe_prob(frame["late_mean"])
    frame["ctw_entropy_bits"] = binary_entropy_bits(p)
    frame["ctw_self_info_bits"] = self_information_bits(p, frame["y_true"])
    frame["ctw_entropy_band"] = entropy_band(frame["ctw_entropy_bits"])
    frame["ctw_brier"] = (p - y) * (p - y)
    frame["ctw_abs_err"] = (p - y).abs()
    frame["ctw_pred_correct"] = ((p >= 0.5).astype("int8") == frame["y_true"].astype("int8")).astype("int8")
    frame["ctw_late_mean"] = p
    return frame


def finalize_anchor_band(grouped):
    grouped = grouped.rename(
        columns={
            "y_true": "empirical_acc",
            "late_mean": "ctw_mean_prob_raw",
            "ctw_late_mean": "ctw_mean_prob",
            "ctw_entropy_bits": "mean_ctw_entropy_bits",
            "ctw_self_info_bits": "mean_ctw_self_info_bits",
            "ctw_brier": "ctw_brier",
            "ctw_abs_err": "ctw_mae",
            "ctw_pred_correct": "ctw_acc",
        }
    )
    grouped["ctw_entropy_band_label"] = grouped["ctw_entropy_band"].to_pandas().map(band_label).values
    return grouped


def anchor_and_item_diagnostics(repo: Path, output_dir: Path, datasets: list[str], split: str):
    anchor_rows = []

    for dataset in datasets:
        ctw_path = resolve_ctw_path(repo, dataset, split)
        frame = cudf.read_csv(str(ctw_path))
        frame = add_anchor_columns(frame)
        frame["dataset"] = dataset
        frame["evaluation_split"] = split

        band = (
            frame.groupby(["dataset", "evaluation_split", "ctw_entropy_band"])
            .agg(
                n=("y_true", "count"),
                empirical_acc=("y_true", "mean"),
                ctw_mean_prob=("ctw_late_mean", "mean"),
                mean_ctw_entropy_bits=("ctw_entropy_bits", "mean"),
                mean_ctw_self_info_bits=("ctw_self_info_bits", "mean"),
                ctw_brier=("ctw_brier", "mean"),
                ctw_mae=("ctw_abs_err", "mean"),
                ctw_acc=("ctw_pred_correct", "mean"),
            )
            .reset_index()
        )
        anchor_rows.append(finalize_anchor_band(band).to_pandas())

    anchor_df = pd.concat(anchor_rows, ignore_index=True).sort_values(["dataset", "ctw_entropy_band"])
    anchor_df.to_csv(output_dir / "ctw_entropy_band_anchor_summary.csv", index=False)
    item_df, item_summary_df = item_diagnostics_from_alignment(output_dir, datasets, split)
    item_df.to_csv(output_dir / "ctw_item_ru_ig.csv", index=False)
    item_summary_df.to_csv(output_dir / "ctw_item_ru_ig_summary.csv", index=False)
    return anchor_df, item_summary_df


def item_diagnostics_from_alignment(output_dir: Path, datasets: list[str], split: str):
    manifest = pd.read_csv(output_dir / "manifest.csv")
    manifest = manifest[
        (manifest["status"] == "ok")
        & (manifest["split"] == split)
        & (manifest["dataset"].isin(datasets))
    ].copy()
    representative = manifest.sort_values(["dataset", "fold", "model", "emb_type"]).drop_duplicates(["dataset", "fold"])
    item_frames = []
    item_summary_rows = []

    for dataset in datasets:
        shards = representative[representative["dataset"] == dataset]
        frames = []
        for row in shards.itertuples(index=False):
            frame = cudf.read_parquet(
                row.output_shard,
                columns=["kt_questions", "ctw_y_true", "ctw_late_mean"],
            ).dropna()
            frame["dataset"] = dataset
            frame = frame.rename(columns={"kt_questions": "item_key", "ctw_y_true": "y_true"})
            frame["ctw_late_mean"] = safe_prob(frame["ctw_late_mean"])
            frame["ctw_entropy_bits"] = binary_entropy_bits(frame["ctw_late_mean"])
            y = frame["y_true"].astype("float64")
            frame["ctw_brier"] = (frame["ctw_late_mean"] - y) * (frame["ctw_late_mean"] - y)
            frames.append(frame)
        if not frames:
            continue
        frame = cudf.concat(frames, ignore_index=True)
        item = (
            frame.groupby(["dataset", "item_key"])
            .agg(
                n=("y_true", "count"),
                empirical_acc=("y_true", "mean"),
                ctw_mean_prob=("ctw_late_mean", "mean"),
                ru_ctw_entropy_bits=("ctw_entropy_bits", "mean"),
                ctw_brier=("ctw_brier", "mean"),
            )
            .reset_index()
        )
        item["marginal_entropy_bits"] = binary_entropy_bits(item["empirical_acc"])
        item["ig_ctw_bits"] = item["marginal_entropy_bits"] - item["ru_ctw_entropy_bits"]
        item_frames.append(item.to_pandas())

        negative_ig_rate = float((item["ig_ctw_bits"] < 0).mean())
        high_ru_rate = float((item["ru_ctw_entropy_bits"] >= 0.8).mean())
        high_ig_rate = float((item["ig_ctw_bits"] >= 0.1).mean())
        item_summary_rows.append(
            {
                "dataset": dataset,
                "evaluation_split": split,
                "num_items": int(len(item)),
                "num_predictions": int(item["n"].sum()),
                "mean_ru_ctw_entropy_bits": float(item["ru_ctw_entropy_bits"].mean()),
                "mean_marginal_entropy_bits": float(item["marginal_entropy_bits"].mean()),
                "mean_ig_ctw_bits": float(item["ig_ctw_bits"].mean()),
                "negative_ig_item_rate": negative_ig_rate,
                "high_ru_item_rate_ru_ge_0p8": high_ru_rate,
                "high_ig_item_rate_ig_ge_0p1": high_ig_rate,
                "mean_item_ctw_brier": float(item["ctw_brier"].mean()),
            }
        )

    if item_frames:
        item_df = pd.concat(item_frames, ignore_index=True).sort_values(["dataset", "ig_ctw_bits"], ascending=[True, False])
    else:
        item_df = pd.DataFrame(columns=["dataset", "item_key", "n", "empirical_acc", "ctw_mean_prob", "ru_ctw_entropy_bits", "ctw_brier", "marginal_entropy_bits", "ig_ctw_bits"])
    item_summary_df = pd.DataFrame(item_summary_rows)
    return item_df, item_summary_df


def init_stats():
    return {
        "n": 0,
        "sum_kt_entropy": 0.0,
        "sum_ctw_entropy": 0.0,
        "sum_abs_entropy_gap": 0.0,
        "sum_signed_entropy_gap": 0.0,
        "sum_kt_brier": 0.0,
        "sum_ctw_brier": 0.0,
        "sum_kt_acc": 0.0,
        "sum_ctw_acc": 0.0,
        "sum_y": 0.0,
        "sum_kt_prob": 0.0,
        "sum_ctw_prob": 0.0,
        "sum_ctw_self_info": 0.0,
    }


def add_stat(target, source):
    for key, value in source.items():
        target[key] += value


def frame_stats(frame):
    out = init_stats()
    if len(frame) == 0:
        return out
    y = frame["ctw_y_true"].astype("float64")
    kt = safe_prob(frame["kt_late_mean"])
    ctw = safe_prob(frame["ctw_late_mean"])
    kt_entropy = binary_entropy_bits(kt)
    ctw_entropy = binary_entropy_bits(ctw)
    entropy_gap = kt_entropy - ctw_entropy
    out["n"] = int(len(frame))
    out["sum_kt_entropy"] = float(kt_entropy.sum())
    out["sum_ctw_entropy"] = float(ctw_entropy.sum())
    out["sum_abs_entropy_gap"] = float(entropy_gap.abs().sum())
    out["sum_signed_entropy_gap"] = float(entropy_gap.sum())
    out["sum_kt_brier"] = float(((kt - y) * (kt - y)).sum())
    out["sum_ctw_brier"] = float(((ctw - y) * (ctw - y)).sum())
    out["sum_kt_acc"] = float(((kt >= 0.5).astype("int8") == frame["ctw_y_true"].astype("int8")).sum())
    out["sum_ctw_acc"] = float(((ctw >= 0.5).astype("int8") == frame["ctw_y_true"].astype("int8")).sum())
    out["sum_y"] = float(y.sum())
    out["sum_kt_prob"] = float(kt.sum())
    out["sum_ctw_prob"] = float(ctw.sum())
    out["sum_ctw_self_info"] = float(self_information_bits(ctw, frame["ctw_y_true"]).sum())
    return out


def finalize_stats(row_key, stats):
    n = stats["n"]
    row = dict(row_key)
    row["n"] = n
    if n == 0:
        return row
    row.update(
        {
            "kt_mean_entropy_bits": stats["sum_kt_entropy"] / n,
            "ctw_mean_entropy_bits": stats["sum_ctw_entropy"] / n,
            "mean_abs_entropy_gap_bits": stats["sum_abs_entropy_gap"] / n,
            "mean_signed_entropy_gap_kt_minus_ctw_bits": stats["sum_signed_entropy_gap"] / n,
            "kt_brier": stats["sum_kt_brier"] / n,
            "ctw_brier": stats["sum_ctw_brier"] / n,
            "brier_delta_kt_minus_ctw": (stats["sum_kt_brier"] - stats["sum_ctw_brier"]) / n,
            "kt_acc": stats["sum_kt_acc"] / n,
            "ctw_acc": stats["sum_ctw_acc"] / n,
            "empirical_acc": stats["sum_y"] / n,
            "kt_mean_prob": stats["sum_kt_prob"] / n,
            "ctw_mean_prob": stats["sum_ctw_prob"] / n,
            "ctw_mean_self_info_bits": stats["sum_ctw_self_info"] / n,
        }
    )
    return row


def model_uncertainty_gap(repo: Path, output_dir: Path, split: str):
    manifest = pd.read_csv(output_dir / "manifest.csv")
    manifest = manifest[(manifest["status"] == "ok") & (manifest["split"] == split)]
    overall = {}
    by_band = {}
    by_model_band = {}

    for row in manifest.itertuples(index=False):
        shard = cudf.read_parquet(
            row.output_shard,
            columns=["kt_late_mean", "ctw_late_mean", "ctw_y_true"],
        ).dropna()
        ctw_entropy = binary_entropy_bits(shard["ctw_late_mean"])
        shard["ctw_entropy_band"] = entropy_band(ctw_entropy)

        key = {"dataset": row.dataset, "evaluation_split": split}
        overall.setdefault(tuple(key.items()), init_stats())
        add_stat(overall[tuple(key.items())], frame_stats(shard))

        for band_index in range(5):
            sub = shard[shard["ctw_entropy_band"] == band_index]
            band_key = {
                "dataset": row.dataset,
                "evaluation_split": split,
                "ctw_entropy_band": band_index,
                "ctw_entropy_band_label": band_label(band_index),
            }
            by_band.setdefault(tuple(band_key.items()), init_stats())
            add_stat(by_band[tuple(band_key.items())], frame_stats(sub))

            model_key = {
                "dataset": row.dataset,
                "evaluation_split": split,
                "model": row.model,
                "emb_type": row.emb_type,
                "ctw_entropy_band": band_index,
                "ctw_entropy_band_label": band_label(band_index),
            }
            by_model_band.setdefault(tuple(model_key.items()), init_stats())
            add_stat(by_model_band[tuple(model_key.items())], frame_stats(sub))

    overall_df = pd.DataFrame(finalize_stats(dict(key), stats) for key, stats in overall.items())
    band_df = pd.DataFrame(finalize_stats(dict(key), stats) for key, stats in by_band.items())
    model_band_df = pd.DataFrame(finalize_stats(dict(key), stats) for key, stats in by_model_band.items())

    overall_df.to_csv(output_dir / "kt_ctw_uncertainty_gap_summary.csv", index=False)
    band_df.sort_values(["dataset", "ctw_entropy_band"]).to_csv(output_dir / "kt_ctw_entropy_band_model_perf_summary.csv", index=False)
    model_band_df.sort_values(["dataset", "model", "emb_type", "ctw_entropy_band"]).to_csv(
        output_dir / "kt_ctw_entropy_band_model_perf_by_model.csv", index=False
    )
    return overall_df, band_df


def main():
    parser = argparse.ArgumentParser(description="Build uncertainty-conditioned paper support tables with cuDF.")
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--alignment-dir", type=Path, default=Path("runs/kt_ctw_question_alignment"))
    parser.add_argument("--datasets", nargs="+", default=["nips_task34", "algebra2005", "assist2015"])
    parser.add_argument("--split", choices=["question", "question_window"], default="question_window")
    args = parser.parse_args()
    try:
        global cudf, cp
        import cudf
        import cupy as cp
        global pd
        import pandas as pd
    except ImportError as exc:
        missing = "cuDF/CuPy" if "cudf" not in str(exc).lower() else "cuDF"
        raise SystemExit(
            f"{missing} is required for this GPU/CUDA-only support tool. Install NVIDIA RAPIDS/cuPy, "
            "or use the derived support/RU-IG tables under results/paper_tables/; see README.md"
        ) from exc

    repo = args.repo.resolve()
    output_dir = (repo / args.alignment_dir).resolve()
    anchor_df, item_summary_df = anchor_and_item_diagnostics(repo, output_dir, args.datasets, args.split)
    gap_df, band_df = model_uncertainty_gap(repo, output_dir, args.split)

    print("[anchor]")
    print(anchor_df.to_string(index=False))
    print("[item]")
    print(item_summary_df.to_string(index=False))
    print("[gap]")
    print(gap_df.to_string(index=False))
    print("[model-band]")
    print(band_df.to_string(index=False))


if __name__ == "__main__":
    main()

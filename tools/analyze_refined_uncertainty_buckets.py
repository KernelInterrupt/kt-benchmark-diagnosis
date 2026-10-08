#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__import__("os").environ.get("PYKT_REPO_ROOT", Path(__file__).resolve().parents[1]))
BASE_ALIGN_DIR = REPO_ROOT / "runs" / "kt_ctw_question_alignment"
RESIDUAL_ALIGN_DIR = REPO_ROOT / "runs" / "simplekt_residual_ctw_local_bayes" / "kt_ctw_question_alignment"
OUTPUT_DIR = REPO_ROOT / "runs" / "refined_uncertainty_sensitivity"

DATASETS = ["nips_task34", "algebra2005"]
MODEL_SPECS = [
    {"model_label": "ctw", "model": "ctw", "emb_type": "ctw", "source": "ctw"},
    {"model_label": "ctw+nn", "model": "simplekt_residual", "emb_type": "qid", "source": "residual"},
    {"model_label": "simplekt", "model": "simplekt", "emb_type": "qid", "source": "base"},
    {"model_label": "AKT", "model": "akt", "emb_type": "qid", "source": "base"},
    {"model_label": "stableKT", "model": "stablekt", "emb_type": "qid", "source": "base"},
]
MODEL_LABEL_ORDER = [row["model_label"] for row in MODEL_SPECS]
MODEL_COLOR_MAP = {
    "ctw": "#4C5563",
    "ctw+nn": "#E07A15",
    "simplekt": "#2A9D8F",
    "AKT": "#1D4ED8",
    "stableKT": "#D1495B",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Refined uncertainty bucket sensitivity analysis.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--base-align-dir", type=Path, default=BASE_ALIGN_DIR)
    parser.add_argument("--residual-align-dir", type=Path, default=RESIDUAL_ALIGN_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--num-buckets", type=int, default=5)
    parser.add_argument("--split", choices=["question", "question_window"], default="question_window")
    return parser.parse_args()


def safe_prob(values, eps: float = 1e-7) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.clip(values, eps, 1.0 - eps)


def binary_entropy_bits(prob) -> np.ndarray:
    prob = safe_prob(prob)
    return -(prob * np.log2(prob) + (1.0 - prob) * np.log2(1.0 - prob))


def bucket_label(idx: int, num_buckets: int) -> str:
    return f"Q{idx + 1}/{num_buckets}"


def assign_equal_frequency_buckets(values: pd.Series, num_buckets: int) -> pd.Series:
    ranked = values.rank(method="first")
    bucket = pd.qcut(ranked, q=num_buckets, labels=False)
    return bucket.astype("int64")


def auc_or_nan(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int8)
    score = safe_prob(score)
    n_pos = int(y_true.sum())
    n_total = int(len(y_true))
    n_neg = n_total - n_pos
    if n_pos <= 0 or n_neg <= 0:
        return float("nan")
    ranks = pd.Series(score).rank(method="average").to_numpy(dtype=np.float64)
    rank_sum = float(ranks[y_true == 1].sum())
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def build_manifest_index(manifest_path: Path, split: str) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    return manifest[(manifest["split"] == split) & (manifest["status"] == "ok")].copy()


def select_manifest_row(manifest: pd.DataFrame, dataset: str, model: str, emb_type: str) -> pd.DataFrame:
    rows = manifest[
        (manifest["dataset"] == dataset)
        & (manifest["model"] == model)
        & (manifest["emb_type"] == emb_type)
    ].copy()
    rows = rows.sort_values(["fold", "output_shard"]).drop_duplicates(["fold"], keep="first")
    return rows


def load_anchor_frame(residual_manifest: pd.DataFrame, dataset: str, num_buckets: int) -> pd.DataFrame:
    rows = select_manifest_row(residual_manifest, dataset, "simplekt_residual", "qid")
    if len(rows) == 0:
        raise FileNotFoundError(f"Missing residual aligned shards for {dataset}")

    parts = []
    use_cols = ["fold", "orirow", "qidx", "ctw_y_true", "ctw_late_mean", "kt_late_mean"]
    for row in rows.itertuples(index=False):
        shard = pd.read_parquet(row.output_shard, columns=use_cols).rename(
            columns={
                "ctw_y_true": "y_true",
                "ctw_late_mean": "p_ctw",
                "kt_late_mean": "p_fused",
            }
        )
        parts.append(shard)

    anchor = pd.concat(parts, ignore_index=True)
    anchor = anchor.dropna(subset=["y_true", "p_ctw", "p_fused"]).reset_index(drop=True)
    anchor["dataset"] = dataset
    anchor["p_ctw"] = safe_prob(anchor["p_ctw"])
    anchor["p_fused"] = safe_prob(anchor["p_fused"])
    anchor["u_ctw_bits"] = binary_entropy_bits(anchor["p_ctw"])
    anchor["u_refined_bits"] = binary_entropy_bits(anchor["p_fused"])
    anchor["bucket_ctw_eqfreq"] = assign_equal_frequency_buckets(anchor["u_ctw_bits"], num_buckets)
    anchor["bucket_refined_eqfreq"] = assign_equal_frequency_buckets(anchor["u_refined_bits"], num_buckets)
    anchor["bucket_ctw_label"] = anchor["bucket_ctw_eqfreq"].map(lambda x: bucket_label(int(x), num_buckets))
    anchor["bucket_refined_label"] = anchor["bucket_refined_eqfreq"].map(lambda x: bucket_label(int(x), num_buckets))
    anchor["sample_key"] = (
        anchor["dataset"].astype(str)
        + "::"
        + anchor["fold"].astype(str)
        + "::"
        + anchor["orirow"].astype(str)
        + "::"
        + anchor["qidx"].astype(str)
    )
    dup_count = int(anchor["sample_key"].duplicated().sum())
    if dup_count:
        raise ValueError(f"Duplicate sample keys detected for {dataset}: {dup_count}")
    return anchor


def load_model_frame(
    dataset: str,
    spec: dict,
    anchor: pd.DataFrame,
    base_manifest: pd.DataFrame,
    residual_manifest: pd.DataFrame,
) -> pd.DataFrame:
    if spec["source"] == "ctw":
        model_frame = anchor[["dataset", "fold", "orirow", "qidx", "y_true", "p_ctw"]].rename(columns={"p_ctw": "prob"}).copy()
    else:
        manifest = residual_manifest if spec["source"] == "residual" else base_manifest
        rows = select_manifest_row(manifest, dataset, spec["model"], spec["emb_type"])
        if len(rows) == 0:
            raise FileNotFoundError(f"Missing aligned shards for {dataset} {spec['model']} {spec['emb_type']}")
        parts = []
        use_cols = ["fold", "orirow", "qidx", "ctw_y_true", "kt_late_mean"]
        for row in rows.itertuples(index=False):
            shard = pd.read_parquet(row.output_shard, columns=use_cols).rename(
                columns={
                    "ctw_y_true": "y_true",
                    "kt_late_mean": "prob",
                }
            )
            parts.append(shard)
        model_frame = pd.concat(parts, ignore_index=True)
        model_frame["prob"] = safe_prob(model_frame["prob"])

    merged = anchor[
        [
            "dataset",
            "fold",
            "orirow",
            "qidx",
            "sample_key",
            "y_true",
            "u_ctw_bits",
            "u_refined_bits",
            "bucket_ctw_eqfreq",
            "bucket_ctw_label",
            "bucket_refined_eqfreq",
            "bucket_refined_label",
            "p_ctw",
            "p_fused",
        ]
    ].merge(
        model_frame[["fold", "orirow", "qidx", "prob"]],
        how="inner",
        on=["fold", "orirow", "qidx"],
        validate="one_to_one",
    )
    if len(merged) == 0:
        raise ValueError(f"Empty aligned intersection for {dataset} {spec['model_label']}")
    merged["model_label"] = spec["model_label"]
    merged["model"] = spec["model"]
    merged["emb_type"] = spec["emb_type"]
    merged["aligned_sample_count"] = int(len(merged))
    merged["anchor_sample_count"] = int(len(anchor))
    return merged


def summarize_bucket_metrics(frame: pd.DataFrame, uncertainty_source: str, num_buckets: int) -> pd.DataFrame:
    bucket_col = "bucket_refined_eqfreq" if uncertainty_source == "refined" else "bucket_ctw_eqfreq"
    label_col = "bucket_refined_label" if uncertainty_source == "refined" else "bucket_ctw_label"
    rows = []
    for (dataset, model_label, model, emb_type, bucket_idx, bucket_label_value), sub in frame.groupby(
        ["dataset", "model_label", "model", "emb_type", bucket_col, label_col],
        sort=True,
    ):
        y_true = sub["y_true"].to_numpy(dtype=np.int8)
        prob = sub["prob"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "dataset": dataset,
                "uncertainty_source": uncertainty_source,
                "model_label": model_label,
                "model": model,
                "emb_type": emb_type,
                "bucket_index": int(bucket_idx),
                "bucket_label": bucket_label_value,
                "bucket_count": int(len(sub)),
                "acc": float(((prob >= 0.5).astype(np.int8) == y_true).mean()),
                "brier": float(np.mean((prob - y_true.astype(np.float64)) ** 2)),
                "auc": auc_or_nan(y_true, prob),
                "mean_prob": float(np.mean(prob)),
                "mean_u_ctw_bits": float(np.mean(sub["u_ctw_bits"])),
                "mean_u_refined_bits": float(np.mean(sub["u_refined_bits"])),
            }
        )

    return pd.DataFrame(rows).sort_values(["dataset", "uncertainty_source", "bucket_index", "model_label"])


def build_delta_table(metrics: pd.DataFrame) -> pd.DataFrame:
    ctw = metrics[metrics["model_label"] == "ctw"][
        ["dataset", "uncertainty_source", "bucket_index", "acc", "brier", "auc"]
    ].rename(
        columns={
            "acc": "ctw_acc",
            "brier": "ctw_brier",
            "auc": "ctw_auc",
        }
    )
    delta = metrics.merge(ctw, on=["dataset", "uncertainty_source", "bucket_index"], how="left", validate="many_to_one")
    delta["delta_acc_vs_ctw"] = delta["acc"] - delta["ctw_acc"]
    delta["delta_brier_vs_ctw"] = delta["brier"] - delta["ctw_brier"]
    delta["delta_auc_vs_ctw"] = delta["auc"] - delta["ctw_auc"]
    return delta


def infer_trend_summary(delta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    target_models = [label for label in MODEL_LABEL_ORDER if label != "ctw"]
    for dataset, sub_dataset in delta.groupby("dataset", sort=True):
        for source, sub_source in sub_dataset.groupby("uncertainty_source", sort=True):
            high = sub_source[sub_source["bucket_index"] == sub_source["bucket_index"].max()]
            low = sub_source[sub_source["bucket_index"] == sub_source["bucket_index"].min()]
            mean_delta_high = float(high[high["model_label"].isin(target_models)]["delta_acc_vs_ctw"].mean())
            mean_delta_low = float(low[low["model_label"].isin(target_models)]["delta_acc_vs_ctw"].mean())
            mean_brier_high = float(high[high["model_label"].isin(target_models)]["delta_brier_vs_ctw"].mean())
            mean_brier_low = float(low[low["model_label"].isin(target_models)]["delta_brier_vs_ctw"].mean())

            top_gain_row = (
                sub_source[sub_source["model_label"].isin(target_models)]
                .sort_values(["delta_acc_vs_ctw", "bucket_index"], ascending=[False, False])
                .iloc[0]
            )
            top_brier_row = (
                sub_source[sub_source["model_label"].isin(target_models)]
                .sort_values(["delta_brier_vs_ctw", "bucket_index"], ascending=[True, False])
                .iloc[0]
            )
            rows.append(
                {
                    "dataset": dataset,
                    "uncertainty_source": source,
                    "mean_delta_acc_low_bucket": mean_delta_low,
                    "mean_delta_acc_high_bucket": mean_delta_high,
                    "mean_delta_brier_low_bucket": mean_brier_low,
                    "mean_delta_brier_high_bucket": mean_brier_high,
                    "top_acc_gain_model": top_gain_row["model_label"],
                    "top_acc_gain_bucket": int(top_gain_row["bucket_index"]),
                    "top_acc_gain_value": float(top_gain_row["delta_acc_vs_ctw"]),
                    "top_brier_gain_model": top_brier_row["model_label"],
                    "top_brier_gain_bucket": int(top_brier_row["bucket_index"]),
                    "top_brier_gain_value": float(top_brier_row["delta_brier_vs_ctw"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["dataset", "uncertainty_source"])


def compare_bucketing_strength(delta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    target_models = [label for label in MODEL_LABEL_ORDER if label != "ctw"]
    for dataset, sub in delta[delta["model_label"].isin(target_models)].groupby("dataset", sort=True):
        ctw = sub[sub["uncertainty_source"] == "ctw"]
        refined = sub[sub["uncertainty_source"] == "refined"]
        for metric in ["delta_acc_vs_ctw", "delta_brier_vs_ctw"]:
            ctw_high = float(ctw[ctw["bucket_index"] == ctw["bucket_index"].max()][metric].mean())
            ctw_low = float(ctw[ctw["bucket_index"] == ctw["bucket_index"].min()][metric].mean())
            refined_high = float(refined[refined["bucket_index"] == refined["bucket_index"].max()][metric].mean())
            refined_low = float(refined[refined["bucket_index"] == refined["bucket_index"].min()][metric].mean())
            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "ctw_high_minus_low": ctw_high - ctw_low,
                    "refined_high_minus_low": refined_high - refined_low,
                }
            )
    out = pd.DataFrame(rows)
    out["comparison"] = np.where(
        np.abs(out["refined_high_minus_low"]) > np.abs(out["ctw_high_minus_low"]),
        "stronger",
        "weaker_or_smoother",
    )
    return out.sort_values(["dataset", "metric"])


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill: str, anchor: str | None = None) -> None:
    if anchor is None:
        draw.text(xy, text, font=font, fill=fill)
    else:
        draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def make_multi_panel_line_figure(
    frame: pd.DataFrame,
    output_path: Path,
    metrics: list[str],
    dataset_order: list[str],
    model_order: list[str],
    title_map: dict[str, str],
    subtitle: str,
) -> None:
    width = 1500
    height = 900
    margin_left = 90
    margin_right = 40
    margin_top = 90
    margin_bottom = 65
    col_gap = 40
    row_gap = 50
    legend_height = 45

    panel_w = int((width - margin_left - margin_right - col_gap) / 2)
    panel_h = int((height - margin_top - margin_bottom - legend_height - row_gap) / 2)

    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (width, height), "#FBFAF7")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw_text(draw, (width // 2, 24), subtitle, font, "#111111", anchor="ma")

    legend_x = margin_left
    legend_y = 54
    for model_label in model_order:
        color = MODEL_COLOR_MAP[model_label]
        draw.rectangle([legend_x, legend_y, legend_x + 14, legend_y + 10], fill=color)
        draw_text(draw, (legend_x + 20, legend_y - 2), model_label, font, "#222222")
        legend_x += 100 if len(model_label) < 8 else 118

    for row_idx, dataset in enumerate(dataset_order):
        for col_idx, metric in enumerate(metrics):
            panel_left = margin_left + col_idx * (panel_w + col_gap)
            panel_top = margin_top + legend_height + row_idx * (panel_h + row_gap)
            panel_right = panel_left + panel_w
            panel_bottom = panel_top + panel_h

            draw.rounded_rectangle([panel_left, panel_top, panel_right, panel_bottom], radius=12, outline="#B9C0C8", width=1, fill="#FFFFFF")
            draw_text(draw, (panel_left + 12, panel_top + 10), f"{dataset} | {title_map[metric]}", font, "#111111")

            sub = frame[frame["dataset"] == dataset].copy()
            num_buckets = int(sub["bucket_index"].nunique())
            xmin, xmax = 0, max(num_buckets - 1, 1)

            values = sub[metric].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=np.float64)
            if len(values) == 0:
                continue
            ymin = float(values.min())
            ymax = float(values.max())
            if metric == "acc":
                ymin = min(ymin, 0.0)
                ymax = max(ymax, 1.0)
            elif metric == "brier":
                ymin = min(ymin, 0.0)
            pad = max((ymax - ymin) * 0.08, 0.01)
            ymin -= pad
            ymax += pad
            if metric.startswith("delta_"):
                ymin = min(ymin, -0.01)
                ymax = max(ymax, 0.01)

            plot_left = panel_left + 50
            plot_right = panel_right - 18
            plot_top = panel_top + 35
            plot_bottom = panel_bottom - 35

            for frac in [0.0, 0.5, 1.0]:
                y_value = ymin + frac * (ymax - ymin)
                y_pix = int(plot_bottom - frac * (plot_bottom - plot_top))
                draw.line([plot_left, y_pix, plot_right, y_pix], fill="#E4E8ED", width=1)
                draw_text(draw, (panel_left + 6, y_pix - 6), f"{y_value:.3f}", font, "#56606B")

            if metric.startswith("delta_") and ymin < 0 < ymax:
                zero_frac = (0.0 - ymin) / (ymax - ymin)
                zero_y = int(plot_bottom - zero_frac * (plot_bottom - plot_top))
                draw.line([plot_left, zero_y, plot_right, zero_y], fill="#777777", width=1)

            draw.line([plot_left, plot_bottom, plot_right, plot_bottom], fill="#47505A", width=1)
            draw.line([plot_left, plot_top, plot_left, plot_bottom], fill="#47505A", width=1)

            for bucket_idx in range(num_buckets):
                if xmax == xmin:
                    x_pix = int((plot_left + plot_right) / 2)
                else:
                    x_pix = int(plot_left + (bucket_idx - xmin) * (plot_right - plot_left) / (xmax - xmin))
                draw.line([x_pix, plot_bottom, x_pix, plot_bottom + 4], fill="#47505A", width=1)
                draw_text(draw, (x_pix, plot_bottom + 8), bucket_label(bucket_idx, num_buckets), font, "#56606B", anchor="ma")

            for model_label in model_order:
                cur = sub[sub["model_label"] == model_label].sort_values("bucket_index")
                if len(cur) == 0:
                    continue
                points = []
                for row in cur.itertuples(index=False):
                    x_pix = int(plot_left + (row.bucket_index - xmin) * (plot_right - plot_left) / (xmax - xmin if xmax != xmin else 1))
                    y_pix = int(plot_bottom - (row._asdict()[metric] - ymin) * (plot_bottom - plot_top) / (ymax - ymin if ymax != ymin else 1))
                    points.append((x_pix, y_pix))
                draw.line(points, fill=MODEL_COLOR_MAP[model_label], width=3)
                for x_pix, y_pix in points:
                    draw.ellipse([x_pix - 4, y_pix - 4, x_pix + 4, y_pix + 4], fill=MODEL_COLOR_MAP[model_label], outline="#FFFFFF")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def make_refined_figure(metrics: pd.DataFrame, output_path: Path) -> None:
    refined = metrics[metrics["uncertainty_source"] == "refined"].copy()
    make_multi_panel_line_figure(
        frame=refined,
        output_path=output_path,
        metrics=["acc", "brier"],
        dataset_order=list(refined["dataset"].drop_duplicates()),
        model_order=MODEL_LABEL_ORDER,
        title_map={"acc": "ACC by refined uncertainty", "brier": "Brier by refined uncertainty"},
        subtitle="Refined uncertainty sensitivity: 5 equal-frequency buckets",
    )


def make_delta_figure(delta: pd.DataFrame, output_path: Path) -> None:
    refined = delta[(delta["uncertainty_source"] == "refined") & (delta["model_label"] != "ctw")].copy()
    make_multi_panel_line_figure(
        frame=refined,
        output_path=output_path,
        metrics=["delta_acc_vs_ctw", "delta_brier_vs_ctw"],
        dataset_order=list(refined["dataset"].drop_duplicates()),
        model_order=[label for label in MODEL_LABEL_ORDER if label != "ctw"],
        title_map={
            "delta_acc_vs_ctw": "Delta ACC vs CTW",
            "delta_brier_vs_ctw": "Delta Brier vs CTW",
        },
        subtitle="Refined uncertainty sensitivity: gains relative to CTW",
    )


def main():
    args = parse_args()
    base_manifest = build_manifest_index(args.base_align_dir / "manifest.csv", args.split)
    residual_manifest = build_manifest_index(args.residual_align_dir / "manifest.csv", args.split)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    anchor_frames = []
    model_frames = []
    coverage_rows = []

    for dataset in args.datasets:
        anchor = load_anchor_frame(residual_manifest, dataset, args.num_buckets)
        anchor_frames.append(anchor)
        coverage_rows.append(
            {
                "dataset": dataset,
                "n_samples": int(len(anchor)),
                "num_folds": int(anchor["fold"].nunique()),
                "u_formula": "H(p_fused)",
                "u_version": "single-model refined proxy",
                "num_buckets": int(args.num_buckets),
                "evaluation_split": args.split,
            }
        )
        for spec in MODEL_SPECS:
            model_frames.append(load_model_frame(dataset, spec, anchor, base_manifest, residual_manifest))

    all_anchor = pd.concat(anchor_frames, ignore_index=True)
    all_models = pd.concat(model_frames, ignore_index=True)

    metrics_frames = [
        summarize_bucket_metrics(all_models, uncertainty_source="ctw", num_buckets=args.num_buckets),
        summarize_bucket_metrics(all_models, uncertainty_source="refined", num_buckets=args.num_buckets),
    ]
    metrics = pd.concat(metrics_frames, ignore_index=True)
    metrics["evaluation_split"] = args.split
    delta = build_delta_table(metrics)
    delta["evaluation_split"] = args.split
    trend_summary = infer_trend_summary(delta)
    trend_summary["evaluation_split"] = args.split
    strength_summary = compare_bucketing_strength(delta)
    strength_summary["evaluation_split"] = args.split

    bucket_counts = (
        all_anchor.groupby(["dataset", "bucket_refined_eqfreq", "bucket_refined_label"], sort=True)
        .size()
        .reset_index(name="bucket_count")
        .rename(columns={"bucket_refined_eqfreq": "bucket_index", "bucket_refined_label": "bucket_label"})
        .sort_values(["dataset", "bucket_index"])
    )
    bucket_counts["evaluation_split"] = args.split

    metrics_path = args.output_dir / "refined_uncertainty_bucket_metrics.csv"
    delta_path = args.output_dir / "refined_uncertainty_bucket_delta_vs_ctw.csv"
    anchor_path = args.output_dir / "refined_uncertainty_anchor_samples.parquet"
    coverage_path = args.output_dir / "analysis_coverage.csv"
    trend_path = args.output_dir / "trend_summary.csv"
    strength_path = args.output_dir / "bucketing_strength_comparison.csv"
    bucket_counts_path = args.output_dir / "refined_bucket_counts.csv"
    fig_path = args.output_dir / "refined_uncertainty_bucket_metrics.png"
    delta_fig_path = args.output_dir / "refined_uncertainty_bucket_delta_vs_ctw.png"
    note_path = args.output_dir / "analysis_notes.json"

    metrics.to_csv(metrics_path, index=False)
    delta.to_csv(delta_path, index=False)
    all_anchor.to_parquet(anchor_path, index=False)
    pd.DataFrame(coverage_rows).to_csv(coverage_path, index=False)
    trend_summary.to_csv(trend_path, index=False)
    strength_summary.to_csv(strength_path, index=False)
    bucket_counts.to_csv(bucket_counts_path, index=False)
    make_refined_figure(metrics, fig_path)
    make_delta_figure(delta, delta_fig_path)

    note_path.write_text(
        json.dumps(
            {
                "u_formula": "u_hat = H(p_fused)",
                "u_version": "single-model refined proxy",
                "ensemble_available": False,
                "ensemble_note": "Only one retained fused prediction per sample/fold was available in local aligned outputs.",
                "bucket_scheme": f"{args.num_buckets} equal-frequency buckets per dataset",
                "evaluation_split": args.split,
                "datasets": list(args.datasets),
                "models": MODEL_LABEL_ORDER,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[done] metrics={metrics_path}")
    print(f"[done] delta={delta_path}")
    print(f"[done] anchor={anchor_path}")
    print(f"[done] coverage={coverage_path}")
    print(f"[done] trend={trend_path}")
    print(f"[done] strength={strength_path}")
    print(f"[done] bucket_counts={bucket_counts_path}")
    print(f"[done] figure={fig_path}")
    print(f"[done] delta_figure={delta_fig_path}")
    print(f"[done] notes={note_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__import__("os").environ.get("PYKT_REPO_ROOT", Path(__file__).resolve().parents[1]))
STRICT_RESIDUAL_EVAL_CSV = REPO_ROOT / "runs" / "strict_residual_bayes_sweeps" / "best_ckpt_test_eval.csv"
BASE_ALIGN_MANIFEST = REPO_ROOT / "runs" / "kt_ctw_question_alignment" / "manifest.csv"
OUTPUT_DIR = REPO_ROOT / "runs" / "paper_exports_uncertainty_conditioned_strict_nips"
STRICT_CTW_PATHS = {
    "question": REPO_ROOT / "runs" / "symbolic_hybrid_benchmark_parallel" / "strict_question_predictions" / "ctw" / "question_predictions.csv",
    "question_window": REPO_ROOT / "runs" / "symbolic_hybrid_benchmark_parallel" / "strict_question_predictions" / "ctw" / "question_window_predictions.csv",
}
SPLIT_TO_PRED_FILE = {
    "question": "qid_test_question_predictions.txt",
    "question_window": "qid_test_question_window_predictions.txt",
}
SPLIT_TO_EVAL_OVERRIDE_COL = {
    "question": "test_question_file_override",
    "question_window": "test_question_window_file_override",
}

MODEL_LABEL_MAP = {
    ("sparsekt", "qid_accumulative_attn"): "sparsekt-soft",
    ("sparsekt", "qid_sparseattn"): "sparsekt-topk",
}

BASE_COLOR_MAP = {
    "simplekt": "#2A9D8F",
    "AKT": "#1D4ED8",
    "stableKT": "#D1495B",
}
COLOR_MAP = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export strict leak-free NIPS uncertainty-band data in paper-table format.")
    parser.add_argument("--strict-ctw-path", type=Path, default=None)
    parser.add_argument("--strict-residual-eval-csv", type=Path, default=STRICT_RESIDUAL_EVAL_CSV)
    parser.add_argument("--base-align-manifest", type=Path, default=BASE_ALIGN_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--strict-method-label", type=str, default="ctw")
    parser.add_argument("--split", choices=["question", "question_window"], default="question_window")
    return parser.parse_args()


def normalize_model_label(model: str, emb_type: str) -> str:
    return MODEL_LABEL_MAP.get((model, emb_type), model)


def anchor_model_label(strict_method_label: str) -> str:
    return str(strict_method_label).strip().lower()


def fused_model_label(strict_method_label: str) -> str:
    return f"{anchor_model_label(strict_method_label)}+nn"


def model_family(model_label: str) -> str:
    if model_label.endswith("+nn"):
        return model_label
        return model_label
    return "kt"


def color_for_model_label(model_label: str, strict_method_label: str) -> str:
    anchor_label = anchor_model_label(strict_method_label)
    if model_label == anchor_label:
        return "#4C5563"
    if model_label == fused_model_label(strict_method_label):
        return "#E07A15"
    return BASE_COLOR_MAP.get(model_label, "#111111")


def safe_prob(values, eps: float = 1e-7) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.clip(values, eps, 1.0 - eps)


def binary_entropy_bits(prob) -> np.ndarray:
    prob = safe_prob(prob)
    return -(prob * np.log2(prob) + (1.0 - prob) * np.log2(1.0 - prob))


def entropy_band(entropy) -> np.ndarray:
    band = np.floor(np.asarray(entropy, dtype=np.float64) * 5.0).astype(np.int64)
    band = np.clip(band, 0, 4)
    return band


def band_label(index: int) -> str:
    return f"{index / 5:.1f}-{(index + 1) / 5:.1f}"


def resolve_strict_ctw_path(split: str, override: Path | None) -> Path:
    if override is not None:
        return override
    return STRICT_CTW_PATHS[split]


def load_strict_anchor(strict_ctw_path: Path, strict_residual_eval_csv: Path, split: str, strict_method_label: str) -> pd.DataFrame:
    eval_col = SPLIT_TO_EVAL_OVERRIDE_COL[split]
    pred_name = SPLIT_TO_PRED_FILE[split]
    ctw = pd.read_csv(strict_ctw_path, usecols=["fold", "qidx", "orirow", "y_true", "late_mean"])
    ctw = ctw.drop_duplicates(["fold", "orirow", "qidx"], keep="first").rename(columns={"late_mean": "p_ctw"})

    eval_df = pd.read_csv(strict_residual_eval_csv)
    eval_df = eval_df[
        (eval_df["dataset_name"] == "nips_task34")
        & (eval_df[eval_col].fillna("").astype(str).str.contains(f"/{strict_method_label}_fold", regex=False))
    ].sort_values("fold")
    if len(eval_df) != 5:
        raise ValueError(f"Expected 5 strict {strict_method_label.upper()} residual folds on split={split}, got {len(eval_df)}")

    parts = []
    for row in eval_df.itertuples(index=False):
        pred_path = Path(row.save_dir) / pred_name
        pred = pd.read_csv(pred_path, sep="\t")[["orirow", "qidx", "late_trues", "late_mean"]].rename(
            columns={"late_trues": "y_true_res", "late_mean": "p_fused"}
        )
        pred["fold"] = int(row.fold)
        parts.append(pred)
    residual = pd.concat(parts, ignore_index=True)

    anchor = ctw.merge(residual, on=["fold", "orirow", "qidx"], how="inner", validate="one_to_one")
    if int((anchor["y_true"] != anchor["y_true_res"]).sum()) != 0:
        raise ValueError(f"Strict {strict_method_label.upper()} / residual labels mismatch")
    anchor = anchor.drop(columns=["y_true_res"])
    anchor["dataset"] = "nips_task34"
    anchor["p_ctw"] = safe_prob(anchor["p_ctw"])
    anchor["p_fused"] = safe_prob(anchor["p_fused"])
    anchor["ctw_entropy_bits"] = binary_entropy_bits(anchor["p_ctw"])
    anchor["ctw_entropy_band"] = entropy_band(anchor["ctw_entropy_bits"])
    anchor["ctw_entropy_band_label"] = [band_label(int(x)) for x in anchor["ctw_entropy_band"]]
    return anchor


def load_base_model_predictions(manifest_path: Path, anchor_keys: pd.DataFrame, split: str) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    manifest = manifest[
        (manifest["dataset"] == "nips_task34")
        & (manifest["split"] == split)
        & (manifest["status"] == "ok")
    ].copy()
    rows = manifest.sort_values(["model", "emb_type", "fold"]).drop_duplicates(["model", "emb_type", "fold"], keep="first")
    parts = []
    for row in rows.itertuples(index=False):
        shard = pd.read_parquet(row.output_shard, columns=["fold", "orirow", "qidx", "kt_late_mean"])
        shard["model"] = row.model
        shard["emb_type"] = row.emb_type
        parts.append(shard)
    preds = pd.concat(parts, ignore_index=True)
    preds["prob"] = safe_prob(preds["kt_late_mean"])
    merged = anchor_keys.merge(preds, on=["fold", "orirow", "qidx"], how="left")
    if int(merged["prob"].isna().sum()) != 0:
        bad = merged[merged["prob"].isna()][["fold", "orirow", "qidx"]].head(5).to_dict("records")
        raise ValueError(f"Missing KT predictions after strict alignment: {bad}")
    return merged.drop(columns=["kt_late_mean"])


def score_rows(frame: pd.DataFrame, prob_col: str, model: str, emb_type: str, model_label: str, split: str) -> list[dict]:
    out = []
    for band_idx, sub in frame.groupby("ctw_entropy_band", sort=True):
        y = sub["y_true"].to_numpy(dtype=np.int8)
        p = safe_prob(sub[prob_col].to_numpy(dtype=np.float64))
        ctw_p = safe_prob(sub["p_ctw"].to_numpy(dtype=np.float64))
        out.append(
            {
                "dataset": "nips_task34",
                "evaluation_split": split,
                "model_family": model_family(model_label),
                "model_label": model_label,
                "model": model,
                "emb_type": emb_type,
                "ctw_entropy_band": int(band_idx),
                "ctw_entropy_band_label": band_label(int(band_idx)),
                "n": int(len(sub)),
                "acc": float(((p >= 0.5).astype(np.int8) == y).mean()),
                "brier": float(np.mean((p - y.astype(np.float64)) ** 2)),
                "mean_prob": float(np.mean(p)),
                "mean_entropy_bits": float(np.mean(binary_entropy_bits(p))),
                "ctw_acc": float(((ctw_p >= 0.5).astype(np.int8) == y).mean()),
                "ctw_brier": float(np.mean((ctw_p - y.astype(np.float64)) ** 2)),
                "ctw_mean_prob": float(np.mean(ctw_p)),
                "ctw_mean_entropy_bits": float(np.mean(binary_entropy_bits(ctw_p))),
                "mean_abs_entropy_gap_bits": float(np.mean(np.abs(binary_entropy_bits(p) - binary_entropy_bits(ctw_p)))),
                "mean_signed_entropy_gap_kt_minus_ctw_bits": float(np.mean(binary_entropy_bits(p) - binary_entropy_bits(ctw_p))),
            }
        )
    return out


def build_band_table(anchor: pd.DataFrame, manifest_path: Path, split: str, strict_method_label: str) -> pd.DataFrame:
    rows = []
    anchor_label = anchor_model_label(strict_method_label)
    rows.extend(score_rows(anchor, "p_ctw", anchor_label, anchor_label, anchor_label, split))
    rows.extend(score_rows(anchor, "p_fused", "simplekt_residual", "qid", fused_model_label(strict_method_label), split))

    anchor_keys = anchor[["fold", "orirow", "qidx"]].copy()
    base_preds = load_base_model_predictions(manifest_path, anchor_keys, split)
    for (model, emb_type), sub in base_preds.groupby(["model", "emb_type"], sort=True):
        aligned = anchor.merge(sub[["fold", "orirow", "qidx", "prob"]], on=["fold", "orirow", "qidx"], how="inner", validate="one_to_one")
        rows.extend(score_rows(aligned, "prob", model, emb_type, normalize_model_label(model, emb_type), split))

    band = pd.DataFrame(rows)
    band["rank_acc_desc_within_band"] = band.groupby(["dataset", "ctw_entropy_band"])["acc"].rank(method="min", ascending=False)
    band["rank_brier_asc_within_band"] = band.groupby(["dataset", "ctw_entropy_band"])["brier"].rank(method="min", ascending=True)
    return band.sort_values(["dataset", "ctw_entropy_band", "rank_acc_desc_within_band", "model_label"]).reset_index(drop=True)


def build_delta_table(band: pd.DataFrame, strict_method_label: str) -> pd.DataFrame:
    anchor_label = anchor_model_label(strict_method_label)
    ctw = band[band["model_label"] == anchor_label][
        ["ctw_entropy_band", "acc", "brier", "mean_prob", "mean_entropy_bits"]
    ].rename(
        columns={
            "acc": "ctw_acc_ref",
            "brier": "ctw_brier_ref",
            "mean_prob": "ctw_mean_prob_ref",
            "mean_entropy_bits": "ctw_mean_entropy_bits_ref",
        }
    )
    delta = band.merge(ctw, on="ctw_entropy_band", how="left", validate="many_to_one")
    delta["delta_acc_vs_ctw"] = delta["acc"] - delta["ctw_acc_ref"]
    delta["delta_brier_vs_ctw"] = delta["brier"] - delta["ctw_brier_ref"]
    return delta


def draw_line_chart(frame: pd.DataFrame, models: list[str], value_col: str, title: str, out_path: Path) -> None:
    width, height = 1200, 700
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (width, height), "#FBFAF7")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    margin_left, margin_right = 90, 40
    margin_top, margin_bottom = 70, 70
    plot_left, plot_right = margin_left, width - margin_right
    plot_top, plot_bottom = margin_top, height - margin_bottom

    draw.text((width // 2, 20), title, font=font, fill="#111111", anchor="ma")

    values = frame[value_col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=np.float64)
    ymin, ymax = float(values.min()), float(values.max())
    if value_col.startswith("delta_"):
        ymin = min(ymin, -0.01)
        ymax = max(ymax, 0.01)
    pad = max((ymax - ymin) * 0.08, 0.01)
    ymin -= pad
    ymax += pad

    for frac in [0.0, 0.5, 1.0]:
        y_val = ymin + frac * (ymax - ymin)
        y_pix = int(plot_bottom - frac * (plot_bottom - plot_top))
        draw.line([plot_left, y_pix, plot_right, y_pix], fill="#E4E8ED", width=1)
        draw.text((12, y_pix - 6), f"{y_val:.3f}", font=font, fill="#56606B")
    if ymin < 0 < ymax:
        zero_frac = (0.0 - ymin) / (ymax - ymin)
        zero_y = int(plot_bottom - zero_frac * (plot_bottom - plot_top))
        draw.line([plot_left, zero_y, plot_right, zero_y], fill="#777777", width=1)

    draw.line([plot_left, plot_bottom, plot_right, plot_bottom], fill="#47505A", width=1)
    draw.line([plot_left, plot_top, plot_left, plot_bottom], fill="#47505A", width=1)

    buckets = sorted(frame["ctw_entropy_band"].unique())
    for idx, bucket in enumerate(buckets):
        x = int(plot_left + idx * (plot_right - plot_left) / max(len(buckets) - 1, 1))
        draw.line([x, plot_bottom, x, plot_bottom + 4], fill="#47505A", width=1)
        draw.text((x, plot_bottom + 10), band_label(int(bucket)), font=font, fill="#56606B", anchor="ma")

    legend_x, legend_y = plot_left, 45
    for model_label in models:
        draw.rectangle([legend_x, legend_y, legend_x + 14, legend_y + 10], fill=COLOR_MAP[model_label])
        draw.text((legend_x + 20, legend_y - 2), model_label, font=font, fill="#222222")
        legend_x += 112

    for model_label in models:
        sub = frame[frame["model_label"] == model_label].sort_values("ctw_entropy_band")
        pts = []
        for idx, row in enumerate(sub.itertuples(index=False)):
            x = int(plot_left + idx * (plot_right - plot_left) / max(len(buckets) - 1, 1))
            y = int(plot_bottom - (getattr(row, value_col) - ymin) * (plot_bottom - plot_top) / max(ymax - ymin, 1e-9))
            pts.append((x, y))
        draw.line(pts, fill=COLOR_MAP[model_label], width=3)
        for x, y in pts:
            draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=COLOR_MAP[model_label], outline="#FFFFFF")

    image.save(out_path)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    strict_ctw_path = resolve_strict_ctw_path(args.split, args.strict_ctw_path)
    anchor = load_strict_anchor(strict_ctw_path, args.strict_residual_eval_csv, args.split, args.strict_method_label)
    band = build_band_table(anchor, args.base_align_manifest, args.split, args.strict_method_label)
    delta = build_delta_table(band, args.strict_method_label)
    delta["evaluation_split"] = args.split
    strict_anchor_rows_before_intersection = int(pd.read_csv(strict_ctw_path, usecols=["fold"]).shape[0])

    band.to_csv(args.output_dir / "question_level_all_models_by_band.csv", index=False)
    delta.to_csv(args.output_dir / "question_level_all_models_delta_by_band.csv", index=False)
    anchor["evaluation_split"] = args.split
    anchor[["evaluation_split", "fold", "orirow", "qidx", "y_true", "p_ctw", "p_fused", "ctw_entropy_bits", "ctw_entropy_band", "ctw_entropy_band_label"]].to_csv(
        args.output_dir / "strict_nips_anchor_question_level.csv", index=False
    )

    coverage_rows = []
    for model_label, sub in band.groupby("model_label", sort=True):
        coverage_rows.append(
            {
                "dataset": "nips_task34",
                "model_label": model_label,
                "model_family": sub["model_family"].iloc[0],
                "evaluation_split": args.split,
                "bands": int(sub["ctw_entropy_band"].nunique()),
                "total_n": int(sub["n"].sum()),
            }
        )
    pd.DataFrame(coverage_rows).sort_values(["model_family", "model_label"]).to_csv(
        args.output_dir / "coverage_summary.csv", index=False
    )

    notes = {
        "dataset": "nips_task34",
        "anchor": f"strict symbolic {args.strict_method_label.upper()} {args.split} predictions",
        "ctw_nn": f"strict leak-free {args.strict_method_label.upper()}+NN best checkpoints",
        "kt_baselines": f"existing aligned {args.split} KT predictions intersected with strict anchor keys",
        "evaluation_split": args.split,
        "band_definition": f"fixed {args.strict_method_label.upper()} anchor entropy bands: floor(H(p_anchor)*5), clipped to 0..4",
        "band_labels": [band_label(i) for i in range(5)],
        "strict_anchor_rows_before_intersection": strict_anchor_rows_before_intersection,
        "strict_rows_after_intersection": int(len(anchor)),
    }
    (args.output_dir / "notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")

    highlight_models = [
        anchor_model_label(args.strict_method_label),
        fused_model_label(args.strict_method_label),
        "simplekt",
        "AKT",
        "stableKT",
    ]
    global COLOR_MAP
    COLOR_MAP = {model_label: color_for_model_label(model_label, args.strict_method_label) for model_label in highlight_models}
    focus_band = band[band["model_label"].isin(highlight_models)].copy()
    focus_delta = delta[delta["model_label"].isin([x for x in highlight_models if x != anchor_model_label(args.strict_method_label)])].copy()
    draw_line_chart(
        focus_band,
        highlight_models,
        "acc",
        f"Strict NIPS uncertainty band accuracy ({args.strict_method_label.upper()} anchor)",
        args.output_dir / "uncertainty_band_accuracy.png",
    )
    draw_line_chart(
        focus_delta,
        [x for x in highlight_models if x != anchor_model_label(args.strict_method_label)],
        "delta_acc_vs_ctw",
        f"Strict NIPS uncertainty band delta accuracy vs {args.strict_method_label.upper()}",
        args.output_dir / "uncertainty_band_delta_accuracy.png",
    )

    readme = [
        "# Strict NIPS Uncertainty Band Export",
        "",
        f"- Split: `{args.split}`",
        "- `question_level_all_models_by_band.csv`: Main band table in the same schema as the old paper export.",
        "- `question_level_all_models_delta_by_band.csv`: Delta table relative to CTW.",
        "- `strict_nips_anchor_question_level.csv`: Strict sample-level anchor with CTW / fused probabilities.",
        "- `uncertainty_band_accuracy.png`: Lightweight ACC band plot for highlighted models.",
        "- `uncertainty_band_delta_accuracy.png`: Lightweight delta-ACC plot vs CTW.",
    ]
    (args.output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"[done] wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from sklearn.metrics import roc_auc_score


REPO_ROOT = Path(__import__("os").environ.get("PYKT_REPO_ROOT", Path(__file__).resolve().parents[1]))
BASE_ALIGN_DIR = REPO_ROOT / "runs" / "kt_ctw_question_alignment"
RESIDUAL_ALIGN_DIR = REPO_ROOT / "runs" / "simplekt_residual_ctw_local_bayes" / "kt_ctw_question_alignment"
RESIDUAL_SWEEP_DIR = REPO_ROOT / "runs" / "simplekt_residual_ctw_local_bayes"
OUTPUT_DIR = REPO_ROOT / "runs" / "paper_exports_uncertainty_conditioned"
BEST_ROOT = REPO_ROOT / "runs" / "best_hparams"
SPLIT_TO_RAW_GLOB = {
    "question": "*_test_question_predictions.txt",
    "question_window": "*_test_question_window_predictions.txt",
}
SPLIT_TO_CTW_PATHS = {
    "question": {
        "nips_task34": "runs/ctw_benchmark_5fold/question_predictions.csv",
        "algebra2005": "runs/ctw_benchmark_algebra2005_5fold/question_predictions.csv",
        "assist2015": "runs/symbolic_hybrid_benchmark_parallel/strict_question_predictions_assist2015/ctw/question_predictions.csv",
    },
    "question_window": {
        "nips_task34": "runs/symbolic_hybrid_benchmark_parallel/strict_question_predictions/ctw/question_window_predictions.csv",
        "algebra2005": "runs/symbolic_hybrid_benchmark_parallel/strict_question_predictions_algebra2005/ctw/question_window_predictions.csv",
        "assist2015": "runs/symbolic_hybrid_benchmark_parallel/strict_question_predictions_assist2015/ctw/question_window_predictions.csv",
    },
}


MODEL_LABEL_MAP = {
    ("sparsekt", "qid_accumulative_attn"): "sparsekt-soft",
    ("sparsekt", "qid_sparseattn"): "sparsekt-topk",
    ("simplekt_residual", "qid"): "ctw+nn",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Export full paper tables for KT + CTW + CTW+NN.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--base-align-dir", type=Path, default=BASE_ALIGN_DIR)
    parser.add_argument("--residual-align-dir", type=Path, default=RESIDUAL_ALIGN_DIR)
    parser.add_argument("--residual-sweep-dir", type=Path, default=RESIDUAL_SWEEP_DIR)
    parser.add_argument("--best-root", type=Path, default=BEST_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--split", choices=["question", "question_window"], default="question_window")
    parser.add_argument("--check-inputs", action="store_true", help="Report missing inputs and exit.")
    return parser.parse_args()


def check_inputs(args) -> int:
    paths = [args.base_align_dir, args.residual_align_dir, args.residual_sweep_dir, args.best_root]
    paths += [REPO_ROOT / rel for rel in SPLIT_TO_CTW_PATHS[args.split].values()]
    missing = [Path(path) for path in paths if not Path(path).exists()]
    if missing:
        print("Missing required inputs (see README.md 'Expected input layout'; override the --*-dir options):")
        for path in missing:
            print(f"- {path}")
    else:
        print("All default required inputs are present.")
    return 0


def normalize_model_label(model: str, emb_type: str) -> str:
    return MODEL_LABEL_MAP.get((model, emb_type), model)


def model_family(model_label: str) -> str:
    if model_label == "ctw":
        return "ctw"
    if model_label == "ctw+nn":
        return "ctw+nn"
    return "kt"


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def safe_prob(values, eps: float = 1e-7):
    values = np.asarray(values, dtype=np.float64)
    return np.clip(values, eps, 1.0 - eps)


def binary_entropy_bits(prob):
    prob = safe_prob(prob)
    return -(prob * np.log2(prob) + (1.0 - prob) * np.log2(1.0 - prob))


def self_information_bits(prob, truth):
    prob = safe_prob(prob)
    truth = np.asarray(truth, dtype=np.int8)
    true_prob = np.where(truth == 1, prob, 1.0 - prob)
    return -np.log2(safe_prob(true_prob))


def score_stats(y_true, prob) -> dict:
    y_true = np.asarray(y_true, dtype=np.int8)
    prob = safe_prob(prob)
    if len(y_true) == 0:
        return {
            "n": 0,
            "auc": np.nan,
            "acc": np.nan,
            "brier": np.nan,
            "mean_prob": np.nan,
            "mean_entropy_bits": np.nan,
            "mean_self_info_bits": np.nan,
        }
    entropy = binary_entropy_bits(prob)
    out = {
        "n": int(len(y_true)),
        "auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else np.nan,
        "acc": float(((prob >= 0.5).astype(np.int8) == y_true).mean()),
        "brier": float(np.mean((prob - y_true.astype(np.float64)) ** 2)),
        "mean_prob": float(np.mean(prob)),
        "mean_entropy_bits": float(np.mean(entropy)),
        "mean_self_info_bits": float(np.mean(self_information_bits(prob, y_true))),
    }
    return out


def build_overall_question_tables(base_dir: Path, residual_dir: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_manifest = load_csv(base_dir / "manifest.csv")
    residual_manifest = load_csv(residual_dir / "manifest.csv")
    base_question = base_manifest[(base_manifest["split"] == split) & (base_manifest["status"] == "ok")].copy()
    residual_question = residual_manifest[
        (residual_manifest["split"] == split) & (residual_manifest["status"] == "ok")
    ].copy()

    rows: list[dict] = []

    def process_manifest(frame: pd.DataFrame, source: str, score_col: str, label_override: str | None = None) -> None:
        for row in frame.itertuples(index=False):
            shard = pd.read_parquet(row.output_shard, columns=["ctw_y_true", score_col]).dropna()
            stats = score_stats(shard["ctw_y_true"].to_numpy(), shard[score_col].to_numpy())
            model_label = label_override or normalize_model_label(row.model, row.emb_type)
            rows.append(
                {
                    "dataset": row.dataset,
                    "fold": int(row.fold),
                    "model": label_override or row.model,
                    "emb_type": "ctw" if label_override == "ctw" else row.emb_type,
                    "model_label": model_label,
                    "model_family": model_family(model_label),
                    "evaluation_split": split,
                    "source": source,
                    **stats,
                }
            )

    process_manifest(base_question, "kt_ctw_question_alignment", "kt_late_mean")
    process_manifest(residual_question, "simplekt_residual_ctw_local_bayes", "kt_late_mean")

    ctw_question = base_question.sort_values(["dataset", "fold", "model", "emb_type"]).drop_duplicates(["dataset", "fold"])
    process_manifest(ctw_question, "ctw_benchmark", "ctw_late_mean", label_override="ctw")

    by_fold = pd.DataFrame(rows)
    by_fold = by_fold.sort_values(["dataset", "model_family", "model_label", "fold"]).reset_index(drop=True)

    summary = (
        by_fold.groupby(["dataset", "model_family", "model_label", "model", "emb_type", "source"], dropna=False)
        .agg(
            folds=("fold", "nunique"),
            n_total=("n", "sum"),
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            acc_mean=("acc", "mean"),
            acc_std=("acc", "std"),
            brier_mean=("brier", "mean"),
            brier_std=("brier", "std"),
            mean_prob_mean=("mean_prob", "mean"),
            mean_entropy_bits_mean=("mean_entropy_bits", "mean"),
            mean_self_info_bits_mean=("mean_self_info_bits", "mean"),
        )
        .reset_index()
    )

    summary["rank_auc_desc"] = summary.groupby("dataset")["auc_mean"].rank(method="min", ascending=False)
    summary["rank_acc_desc"] = summary.groupby("dataset")["acc_mean"].rank(method="min", ascending=False)
    summary["rank_brier_asc"] = summary.groupby("dataset")["brier_mean"].rank(method="min", ascending=True)
    summary = summary.sort_values(["dataset", "rank_auc_desc", "rank_brier_asc", "model_label"]).reset_index(drop=True)
    summary["evaluation_split"] = split
    return by_fold, summary


def build_band_tables(base_dir: Path, residual_dir: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_band = load_csv(base_dir / "kt_ctw_entropy_band_model_perf_by_model.csv")
    residual_band = load_csv(residual_dir / "kt_ctw_entropy_band_model_perf_by_model.csv")
    anchor = load_csv(residual_dir / "ctw_entropy_band_anchor_summary.csv")

    for frame in (base_band, residual_band):
        frame["model_label"] = [normalize_model_label(model, emb_type) for model, emb_type in zip(frame["model"], frame["emb_type"])]
        frame["model_family"] = [model_family(x) for x in frame["model_label"]]
        frame["evaluation_split"] = split
        frame["auc"] = pd.NA
        frame["acc"] = frame["kt_acc"]
        frame["brier"] = frame["kt_brier"]
        frame["mean_prob"] = frame["kt_mean_prob"]
        frame["mean_entropy_bits"] = frame["kt_mean_entropy_bits"]

    base_band = base_band[
        [
            "dataset",
            "evaluation_split",
            "model_family",
            "model_label",
            "model",
            "emb_type",
            "ctw_entropy_band",
            "ctw_entropy_band_label",
            "n",
            "acc",
            "brier",
            "mean_prob",
            "mean_entropy_bits",
            "ctw_acc",
            "ctw_brier",
            "ctw_mean_prob",
            "ctw_mean_entropy_bits",
            "mean_abs_entropy_gap_bits",
            "mean_signed_entropy_gap_kt_minus_ctw_bits",
        ]
    ].copy()
    residual_band = residual_band[
        [
            "dataset",
            "evaluation_split",
            "model_family",
            "model_label",
            "model",
            "emb_type",
            "ctw_entropy_band",
            "ctw_entropy_band_label",
            "n",
            "acc",
            "brier",
            "mean_prob",
            "mean_entropy_bits",
            "ctw_acc",
            "ctw_brier",
            "ctw_mean_prob",
            "ctw_mean_entropy_bits",
            "mean_abs_entropy_gap_bits",
            "mean_signed_entropy_gap_kt_minus_ctw_bits",
        ]
    ].copy()

    anchor_rows = pd.DataFrame(
        {
            "dataset": anchor["dataset"],
            "evaluation_split": split,
            "model_family": "ctw",
            "model_label": "ctw",
            "model": "ctw",
            "emb_type": "ctw",
            "ctw_entropy_band": anchor["ctw_entropy_band"],
            "ctw_entropy_band_label": anchor["ctw_entropy_band_label"],
            "n": anchor["n"],
            "acc": anchor["ctw_acc"],
            "brier": anchor["ctw_brier"],
            "mean_prob": anchor["ctw_mean_prob"],
            "mean_entropy_bits": anchor["mean_ctw_entropy_bits"],
            "ctw_acc": anchor["ctw_acc"],
            "ctw_brier": anchor["ctw_brier"],
            "ctw_mean_prob": anchor["ctw_mean_prob"],
            "ctw_mean_entropy_bits": anchor["mean_ctw_entropy_bits"],
            "mean_abs_entropy_gap_bits": 0.0,
            "mean_signed_entropy_gap_kt_minus_ctw_bits": 0.0,
        }
    )

    band = pd.concat([base_band, anchor_rows, residual_band], ignore_index=True)
    band["rank_acc_desc_within_band"] = band.groupby(["dataset", "ctw_entropy_band"])["acc"].rank(method="min", ascending=False)
    band["rank_brier_asc_within_band"] = band.groupby(["dataset", "ctw_entropy_band"])["brier"].rank(method="min", ascending=True)
    band = band.sort_values(["dataset", "ctw_entropy_band", "rank_acc_desc_within_band", "model_label"]).reset_index(drop=True)

    summary = (
        band.groupby(["dataset", "model_family", "model_label", "model", "emb_type"], dropna=False)
        .agg(
            total_n=("n", "sum"),
            low_acc_mean=("acc", lambda s: s.iloc[0] if len(s) > 0 else pd.NA),
        )
        .reset_index()
    )
    del summary["low_acc_mean"]
    summary["evaluation_split"] = split
    return band, summary


def build_ctw_nn_hparam_table(residual_sweep_dir: Path) -> pd.DataFrame:
    summary_path = residual_sweep_dir / "best_validnll_summary.csv"
    df = load_csv(summary_path)
    keep = [
        "task",
        "dataset_name",
        "fold",
        "seed",
        "save_dir",
        "model_save_path",
        "trial_index",
        "best_epoch",
        "duration_sec",
        "validnll",
        "validauc",
        "validacc",
        "validbrier",
        "validece",
        "train_batch_size",
        "d_model",
        "n_blocks",
        "dropout",
        "learning_rate",
        "final_fc_dim",
        "final_fc_dim2",
        "ctw_feat_dim",
        "ctw_delta_scale",
    ]
    return df[keep].copy()


def build_ru_ig_support_sensitivity(residual_dir: Path, split: str, thresholds=(1, 5, 10, 20, 50, 100)) -> pd.DataFrame:
    item = load_csv(residual_dir / "ctw_item_ru_ig.csv")
    rows = []
    for dataset, sub in item.groupby("dataset"):
        for threshold in thresholds:
            cur = sub[sub["n"] >= threshold].copy()
            if len(cur) == 0:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "evaluation_split": split,
                    "min_item_support": threshold,
                    "num_items": int(len(cur)),
                    "num_predictions": int(cur["n"].sum()),
                    "mean_ru_ctw_entropy_bits": float(cur["ru_ctw_entropy_bits"].mean()),
                    "mean_marginal_entropy_bits": float(cur["marginal_entropy_bits"].mean()),
                    "mean_ig_ctw_bits": float(cur["ig_ctw_bits"].mean()),
                    "negative_ig_item_rate": float((cur["ig_ctw_bits"] < 0).mean()),
                    "high_ru_item_rate_ru_ge_0p8": float((cur["ru_ctw_entropy_bits"] >= 0.8).mean()),
                    "high_ig_item_rate_ig_ge_0p1": float((cur["ig_ctw_bits"] >= 0.1).mean()),
                }
            )
    return pd.DataFrame(rows)


def compute_question_prediction_metrics(path: Path, truth_col: str, score_col: str) -> dict | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(path, sep="\t" if path.suffix == ".txt" else ",")
    except EmptyDataError:
        return None
    if truth_col not in df.columns or score_col not in df.columns:
        return None
    df = df[[truth_col, score_col]].dropna()
    if df.empty:
        return None
    return score_stats(df[truth_col].to_numpy(), df[score_col].to_numpy())


def build_raw_question_tables(best_root: Path, residual_sweep_dir: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    pred_glob = SPLIT_TO_RAW_GLOB[split]

    for config_path in sorted(best_root.rglob("config.json")):
        save_dir = config_path.parent
        pred_files = sorted(save_dir.glob(pred_glob))
        if not pred_files:
            continue
        pred_path = pred_files[0]
        config = read_json(config_path)
        params = config.get("params", {})
        dataset = params.get("dataset_name", "")
        model = params.get("model_name", "")
        emb_type = params.get("emb_type", "")
        fold = params.get("fold")
        if not dataset or not model or fold is None:
            continue
        stats = compute_question_prediction_metrics(pred_path, "late_trues", "late_mean")
        if stats is None:
            continue
        model_label = normalize_model_label(model, emb_type)
        rows.append(
            {
                "dataset": dataset,
                "fold": int(fold),
                "model": model,
                "emb_type": emb_type,
                "model_label": model_label,
                "model_family": model_family(model_label),
                "evaluation_split": split,
                "source": f"best_hparams_raw_{split}",
                **stats,
            }
        )

    residual_summary = load_csv(residual_sweep_dir / "best_validnll_summary.csv")
    for row in residual_summary.itertuples(index=False):
        save_dir = Path(row.save_dir)
        pred_files = sorted(save_dir.glob(pred_glob))
        if not pred_files:
            continue
        pred_path = pred_files[0]
        stats = compute_question_prediction_metrics(pred_path, "late_trues", "late_mean")
        if stats is None:
            continue
        rows.append(
            {
                "dataset": row.dataset_name,
                "fold": int(row.fold),
                "model": "simplekt_residual",
                "emb_type": "qid",
                "model_label": "ctw+nn",
                "model_family": "ctw+nn",
                "evaluation_split": split,
                "source": f"simplekt_residual_raw_{split}",
                **stats,
            }
        )

    for dataset, ctw_rel in SPLIT_TO_CTW_PATHS[split].items():
        ctw_df = pd.read_csv(REPO_ROOT / ctw_rel)
        for fold, sub in ctw_df.groupby("fold"):
            stats = score_stats(sub["y_true"].to_numpy(), sub["late_mean"].to_numpy())
            rows.append(
                {
                    "dataset": dataset,
                    "fold": int(fold),
                    "model": "ctw",
                    "emb_type": "ctw",
                    "model_label": "ctw",
                    "model_family": "ctw",
                    "evaluation_split": split,
                    "source": f"ctw_benchmark_raw_{split}",
                    **stats,
                }
            )

    by_fold = pd.DataFrame(rows).sort_values(["dataset", "model_family", "model_label", "fold"]).reset_index(drop=True)
    summary = (
        by_fold.groupby(["dataset", "model_family", "model_label", "model", "emb_type", "source"], dropna=False)
        .agg(
            folds=("fold", "nunique"),
            n_total=("n", "sum"),
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            acc_mean=("acc", "mean"),
            acc_std=("acc", "std"),
            brier_mean=("brier", "mean"),
            brier_std=("brier", "std"),
            mean_prob_mean=("mean_prob", "mean"),
            mean_entropy_bits_mean=("mean_entropy_bits", "mean"),
            mean_self_info_bits_mean=("mean_self_info_bits", "mean"),
        )
        .reset_index()
    )
    summary["rank_auc_desc"] = summary.groupby("dataset")["auc_mean"].rank(method="min", ascending=False)
    summary["rank_acc_desc"] = summary.groupby("dataset")["acc_mean"].rank(method="min", ascending=False)
    summary["rank_brier_asc"] = summary.groupby("dataset")["brier_mean"].rank(method="min", ascending=True)
    summary = summary.sort_values(["dataset", "rank_auc_desc", "rank_brier_asc", "model_label"]).reset_index(drop=True)
    summary["evaluation_split"] = split
    return by_fold, summary


def export_copy_tables(base_dir: Path, residual_dir: Path, output_dir: Path) -> None:
    copy_map = {
        "ctw_anchor_band_summary.csv": residual_dir / "ctw_entropy_band_anchor_summary.csv",
        "ctw_item_ru_ig.csv": residual_dir / "ctw_item_ru_ig.csv",
        "ctw_item_ru_ig_summary.csv": residual_dir / "ctw_item_ru_ig_summary.csv",
        "ctw_nn_only_overall_gap_summary.csv": residual_dir / "kt_ctw_uncertainty_gap_summary.csv",
        "all_kt_only_overall_gap_summary.csv": base_dir / "kt_ctw_uncertainty_gap_summary.csv",
        "all_kt_only_band_by_model.csv": base_dir / "kt_ctw_entropy_band_model_perf_by_model.csv",
        "ctw_nn_only_band_by_model.csv": residual_dir / "kt_ctw_entropy_band_model_perf_by_model.csv",
    }
    for out_name, src in copy_map.items():
        if src.exists():
            (output_dir / out_name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def write_manifest(output_dir: Path, tables: dict[str, str]) -> None:
    lines = ["# Paper Export Tables", ""]
    for name, desc in tables.items():
        lines.append(f"- `{name}`: {desc}")
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.check_inputs:
        raise SystemExit(check_inputs(args))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_by_fold, overall_mean = build_overall_question_tables(args.base_align_dir, args.residual_align_dir, args.split)
    band_table, _ = build_band_tables(args.base_align_dir, args.residual_align_dir, args.split)
    ctw_nn_hparams = build_ctw_nn_hparam_table(args.residual_sweep_dir)
    ru_ig_support = build_ru_ig_support_sensitivity(args.residual_align_dir, args.split)
    raw_by_fold, raw_mean = build_raw_question_tables(args.best_root, args.residual_sweep_dir, args.split)

    overall_by_fold.to_csv(output_dir / "question_level_all_models_by_fold.csv", index=False)
    overall_mean.to_csv(output_dir / "question_level_all_models_mean.csv", index=False)
    band_table.to_csv(output_dir / "question_level_all_models_by_band.csv", index=False)
    ctw_nn_hparams.to_csv(output_dir / "ctw_nn_best_hparams.csv", index=False)
    ru_ig_support.to_csv(output_dir / "ctw_item_ru_ig_support_sensitivity.csv", index=False)
    raw_by_fold.to_csv(output_dir / "question_level_raw_all_datasets_by_fold.csv", index=False)
    raw_mean.to_csv(output_dir / "question_level_raw_all_datasets_mean.csv", index=False)

    export_copy_tables(args.base_align_dir, args.residual_align_dir, output_dir)

    coverage = (
        overall_by_fold.groupby(["dataset", "model_label", "model_family"], dropna=False)
        .agg(folds=("fold", "nunique"), total_n=("n", "sum"))
        .reset_index()
        .sort_values(["dataset", "model_family", "model_label"])
    )
    coverage["evaluation_split"] = args.split
    coverage.to_csv(output_dir / "coverage_summary.csv", index=False)

    write_manifest(
        output_dir,
        {
            "question_level_all_models_by_fold.csv": f"All {args.split} fold metrics for KT models, CTW, and CTW+NN.",
            "question_level_all_models_mean.csv": f"Fold-averaged {args.split} summary with ranks by AUC/ACC/Brier.",
            "question_level_all_models_by_band.csv": f"Uncertainty-band table for KT models, CTW, and CTW+NN on {args.split}.",
            "ctw_anchor_band_summary.csv": "CTW anchor statistics by entropy band.",
            "ctw_item_ru_ig.csv": f"Item-level RU/IG table derived from CTW anchor on {args.split}.",
            "ctw_item_ru_ig_summary.csv": f"Dataset-level RU/IG summary on {args.split}.",
            "ctw_item_ru_ig_support_sensitivity.csv": f"RU/IG sensitivity to minimum item support on {args.split}.",
            "ctw_nn_best_hparams.csv": "Best hyperparameters for CTW+NN residual model by fold.",
            "question_level_raw_all_datasets_by_fold.csv": f"Raw {args.split} test metrics across all datasets, including assist2015 KT runs.",
            "question_level_raw_all_datasets_mean.csv": f"Fold-averaged raw {args.split} test summary across all datasets.",
            "all_kt_only_band_by_model.csv": "Original all-KT uncertainty-band table without CTW+NN.",
            "ctw_nn_only_band_by_model.csv": "CTW+NN-only uncertainty-band table.",
            "all_kt_only_overall_gap_summary.csv": "Original all-KT overall uncertainty gap summary.",
            "ctw_nn_only_overall_gap_summary.csv": "CTW+NN overall uncertainty gap summary.",
            "coverage_summary.csv": f"Model/dataset coverage and sample counts used in {args.split} exports.",
        },
    )
    print(f"[export] wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

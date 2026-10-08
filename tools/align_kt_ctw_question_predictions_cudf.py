#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import re
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
    "assist2012": {
        "question": "runs/ctw_benchmark_assist2012_5fold/question_predictions.csv",
    },
    "assist2015": {
        "question": "runs/symbolic_hybrid_benchmark_parallel/strict_question_predictions_assist2015/ctw/question_predictions.csv",
        "question_window": "runs/symbolic_hybrid_benchmark_parallel/strict_question_predictions_assist2015/ctw/question_window_predictions.csv",
    },
}

DEFAULT_BLACKLIST = {
    "gkt",
    "skvmn",
    "iekt",
    "qdkt",
    "denoisekt",
    "fluckt",
    "dtransformer",
    "dkvmn",
    "qikt",
    "lpkt",
}

KEY_COLUMNS = ["fold", "orirow", "qidx"]
KT_REQUIRED = ["orirow", "qidx"]
CTW_REQUIRED = ["fold", "orirow", "qidx", "y_true"]
PREDICTION_COLUMNS = [
    "concept_preds",
    "early_preds",
    "late_mean",
    "late_vote",
    "late_all",
]


def parse_csv_set(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(value)).strip("_")


def read_config(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def collect_jobs_from_summary(summary_path: Path, datasets: set[str], blacklist: set[str]) -> list[dict]:
    jobs = []
    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    for row in rows:
        dataset = row.get("dataset_name", "")
        save_dir = Path(row.get("save_dir", ""))
        if not dataset or dataset not in datasets or not save_dir.exists():
            continue
        config = read_config(save_dir / "config.json")
        params = config.get("params", {})
        model = params.get("model_name", "")
        emb_type = params.get("emb_type", "")
        fold = params.get("fold")
        if not model or model in blacklist or fold is None:
            continue

        for split, pattern in [
            ("question", "*_test_question_predictions.txt"),
            ("question_window", "*_test_question_window_predictions.txt"),
        ]:
            files = sorted(save_dir.glob(pattern))
            if not files:
                continue
            run_hash = hashlib.sha1(str(save_dir).encode("utf-8")).hexdigest()[:10]
            run_id = f"{safe_name(dataset)}__{safe_name(model)}__{safe_name(emb_type)}__fold{fold}__{run_hash}"
            jobs.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "emb_type": emb_type,
                    "fold": int(fold),
                    "split": split,
                    "save_dir": str(save_dir),
                    "config_path": str(save_dir / "config.json"),
                    "kt_file": str(files[0]),
                    "run_id": run_id,
                }
            )
    return jobs


def collect_jobs(best_root: Path, datasets: set[str], blacklist: set[str]) -> list[dict]:
    jobs = []
    for config_path in sorted(best_root.rglob("config.json")):
        save_dir = config_path.parent
        config = read_config(config_path)
        params = config.get("params", {})
        dataset = params.get("dataset_name", "")
        model = params.get("model_name", "")
        emb_type = params.get("emb_type", "")
        fold = params.get("fold")
        if not dataset or not model or dataset not in datasets or model in blacklist or fold is None:
            continue

        for split, pattern in [
            ("question", "*_test_question_predictions.txt"),
            ("question_window", "*_test_question_window_predictions.txt"),
        ]:
            files = sorted(save_dir.glob(pattern))
            if not files:
                continue
            run_hash = hashlib.sha1(str(save_dir).encode("utf-8")).hexdigest()[:10]
            run_id = f"{safe_name(dataset)}__{safe_name(model)}__{safe_name(emb_type)}__fold{fold}__{run_hash}"
            jobs.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "emb_type": emb_type,
                    "fold": int(fold),
                    "split": split,
                    "save_dir": str(save_dir),
                    "config_path": str(config_path),
                    "kt_file": str(files[0]),
                    "run_id": run_id,
                }
            )
    return jobs


def prefix_non_keys(frame, prefix: str, keep: set[str]) -> cudf.DataFrame:
    rename_map = {col: f"{prefix}{col}" for col in frame.columns if col not in keep}
    return frame.rename(columns=rename_map)


def numeric_series(frame: cudf.DataFrame, column: str):
    if column not in frame.columns:
        return None
    try:
        return frame[column].astype("float64")
    except Exception:
        return None


def gpu_auc(y_true, score):
    try:
        work = cudf.DataFrame({"y": y_true.astype("int8"), "score": score.astype("float64")}).dropna()
        if len(work) == 0:
            return None
        n_pos = int(work["y"].sum())
        n_total = int(len(work))
        n_neg = n_total - n_pos
        if n_pos <= 0 or n_neg <= 0:
            return None
        ranks = work["score"].rank(method="average")
        rank_sum = float(ranks[work["y"] == 1].sum())
        auc = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        return float(auc)
    except Exception:
        return None


def gpu_acc(y_true, score):
    try:
        work = cudf.DataFrame({"y": y_true.astype("int8"), "score": score.astype("float64")}).dropna()
        if len(work) == 0:
            return None
        pred = (work["score"] >= 0.5).astype("int8")
        return float((pred == work["y"]).mean())
    except Exception:
        return None


def add_metric_columns(metrics: dict, frame: cudf.DataFrame, truth_col: str, prefix: str, pred_prefix: str) -> None:
    if truth_col not in frame.columns:
        return
    y_true = numeric_series(frame, truth_col)
    if y_true is None:
        return
    for col in PREDICTION_COLUMNS:
        full_col = f"{pred_prefix}{col}"
        score = numeric_series(frame, full_col)
        if score is None:
            continue
        metrics[f"{prefix}_{col}_auc"] = gpu_auc(y_true, score)
        metrics[f"{prefix}_{col}_acc"] = gpu_acc(y_true, score)


def prepare_output_path(output_dir: Path, job: dict, output_format: str) -> Path:
    dataset_dir = output_dir / "shards" / safe_name(job["dataset"])
    dataset_dir.mkdir(parents=True, exist_ok=True)
    suffix = "parquet" if output_format == "parquet" else "csv"
    return dataset_dir / f"{job['run_id']}__{job['split']}.{suffix}"


def align_one_job(job: dict, ctw_by_fold: dict[int, cudf.DataFrame], output_dir: Path, output_format: str, overwrite: bool) -> tuple[dict, dict]:
    shard_path = prepare_output_path(output_dir, job, output_format)
    base_manifest = {
        **job,
        "output_shard": str(shard_path),
        "status": "started",
    }
    metrics = {
        "dataset": job["dataset"],
        "model": job["model"],
        "emb_type": job["emb_type"],
        "fold": job["fold"],
        "split": job["split"],
        "run_id": job["run_id"],
    }

    try:
        kt = cudf.read_csv(job["kt_file"], sep="\t")
        missing_kt = [col for col in KT_REQUIRED if col not in kt.columns]
        if missing_kt:
            raise ValueError(f"KT file missing columns: {missing_kt}")
        kt["fold"] = job["fold"]
        kt["run_id"] = job["run_id"]
        kt["dataset"] = job["dataset"]
        kt["model"] = job["model"]
        kt["emb_type"] = job["emb_type"]
        kt["split"] = job["split"]

        ctw = ctw_by_fold.get(job["fold"])
        if ctw is None:
            raise ValueError(f"CTW fold {job['fold']} is unavailable for {job['dataset']}")

        keep = set(KEY_COLUMNS)
        kt_prefixed = prefix_non_keys(kt, "kt_", keep)
        ctw_prefixed = prefix_non_keys(ctw, "ctw_", keep)
        aligned = kt_prefixed.merge(ctw_prefixed, how="left", on=KEY_COLUMNS)

        n_kt = int(len(aligned))
        n_aligned = int(aligned["ctw_y_true"].notnull().sum()) if "ctw_y_true" in aligned.columns else 0
        true_mismatch = None
        if "kt_late_trues" in aligned.columns and "ctw_y_true" in aligned.columns:
            matched = aligned[aligned["ctw_y_true"].notnull()]
            true_mismatch = int((matched["kt_late_trues"].astype("int64") != matched["ctw_y_true"].astype("int64")).sum())

        if overwrite or not shard_path.exists():
            if output_format == "parquet":
                aligned.to_parquet(shard_path, index=False)
            else:
                aligned.to_csv(shard_path, index=False)

        metrics.update(
            {
                "n_kt_rows": n_kt,
                "n_aligned_rows": n_aligned,
                "match_rate": (n_aligned / n_kt) if n_kt else None,
                "true_mismatch_count": true_mismatch,
            }
        )
        add_metric_columns(metrics, aligned, "ctw_y_true", "kt", "kt_")
        add_metric_columns(metrics, aligned, "ctw_y_true", "ctw", "ctw_")

        manifest = {
            **base_manifest,
            "status": "ok",
            "n_kt_rows": n_kt,
            "n_aligned_rows": n_aligned,
            "match_rate": metrics["match_rate"],
            "true_mismatch_count": true_mismatch,
        }
        return manifest, metrics
    except Exception as exc:
        manifest = {
            **base_manifest,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        metrics["error"] = manifest["error"]
        return manifest, metrics


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_ctw_by_fold(ctw_path: Path) -> dict[int, cudf.DataFrame]:
    ctw = cudf.read_csv(str(ctw_path))
    missing = [col for col in CTW_REQUIRED if col not in ctw.columns]
    if missing:
        raise ValueError(f"CTW file {ctw_path} missing columns: {missing}")
    ctw_by_fold = {}
    for fold in sorted(int(x) for x in ctw["fold"].drop_duplicates().to_pandas().tolist()):
        ctw_by_fold[fold] = ctw[ctw["fold"] == fold]
    return ctw_by_fold


def resolve_ctw_path(repo: Path, dataset: str, split: str) -> Path | None:
    cfg = DEFAULT_CTW_PATHS.get(dataset)
    if cfg is None:
        return None
    if isinstance(cfg, dict):
        rel = cfg.get(split) or cfg.get("question")
    else:
        rel = cfg
    if rel is None:
        return None
    return repo / rel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(os.environ.get("PYKT_REPO_ROOT", Path(__file__).resolve().parents[1])))
    parser.add_argument("--best-root", type=Path, default=Path("runs/best_hparams"))
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/kt_ctw_question_alignment"))
    parser.add_argument("--datasets", default="nips_task34,algebra2005,assist2012,assist2015")
    parser.add_argument("--blacklist", default=",".join(sorted(DEFAULT_BLACKLIST)))
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    try:
        global cudf
        import cudf
    except ImportError as exc:
        raise SystemExit(
            "cuDF is required for this GPU/CUDA-only alignment tool. Install NVIDIA RAPIDS cuDF, "
            "or use the derived alignment/model tables under results/paper_tables/; see README.md"
        ) from exc

    repo = args.repo.resolve()
    os.chdir(repo)
    best_root = args.best_root if args.best_root.is_absolute() else repo / args.best_root
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    datasets = parse_csv_set(args.datasets)
    blacklist = parse_csv_set(args.blacklist)

    if args.summary_json is not None:
        summary_path = args.summary_json if args.summary_json.is_absolute() else repo / args.summary_json
        jobs = collect_jobs_from_summary(summary_path, datasets, blacklist)
    else:
        jobs = collect_jobs(best_root, datasets, blacklist)
    if args.limit:
        jobs = jobs[: args.limit]
    jobs_by_dataset = {}
    for job in jobs:
        jobs_by_dataset.setdefault(job["dataset"], []).append(job)

    all_manifest = []
    all_metrics = []
    for dataset in sorted(jobs_by_dataset):
        jobs_by_split = {}
        for job in jobs_by_dataset[dataset]:
            jobs_by_split.setdefault(job["split"], []).append(job)
        for split in sorted(jobs_by_split):
            ctw_path = resolve_ctw_path(repo, dataset, split)
            if ctw_path is None:
                for job in jobs_by_split[split]:
                    all_manifest.append({**job, "status": "failed", "error": f"No CTW path configured for {dataset} split={split}"})
                continue
            if not ctw_path.exists():
                for job in jobs_by_split[split]:
                    all_manifest.append({**job, "status": "failed", "error": f"Missing CTW file: {ctw_path}"})
                continue

            print(f"[align] loading CTW {dataset} split={split}: {ctw_path}", flush=True)
            ctw_by_fold = load_ctw_by_fold(ctw_path)
            print(f"[align] dataset={dataset} split={split} jobs={len(jobs_by_split[split])} folds={sorted(ctw_by_fold)}", flush=True)
            for index, job in enumerate(jobs_by_split[split], start=1):
                manifest, metrics = align_one_job(job, ctw_by_fold, output_dir, args.format, args.overwrite)
                all_manifest.append(manifest)
                all_metrics.append(metrics)
                print(
                    f"[align] {dataset} {index}/{len(jobs_by_split[split])} "
                    f"{job['model']} fold={job['fold']} split={job['split']} status={manifest['status']} "
                    f"matched={manifest.get('n_aligned_rows', '-')}/{manifest.get('n_kt_rows', '-')}",
                    flush=True,
                )

    manifest_path = output_dir / "manifest.csv"
    metrics_path = output_dir / "metrics.csv"
    write_csv(manifest_path, all_manifest)
    write_csv(metrics_path, all_metrics)
    write_csv(repo / "runs" / "kt_ctw_question_alignment_manifest.csv", all_manifest)
    write_csv(repo / "runs" / "kt_ctw_question_alignment_metrics.csv", all_metrics)
    print(f"[align] wrote {manifest_path}")
    print(f"[align] wrote {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

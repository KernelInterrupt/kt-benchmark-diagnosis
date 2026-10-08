#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

try:
    from tools.per_question_common import resident_family_key
except ImportError:
    from per_question_common import resident_family_key


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = REPO_ROOT / "runs" / "simplekt_residual_ctw_local_bayes"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build per-question eval queue from local-bayes best validnll checkpoints."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--queue-dir", type=Path, default=DEFAULT_ROOT / "test_eval_queue")
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_ROOT / "best_validnll_summary.json")
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_ROOT / "best_validnll_summary.csv")
    parser.add_argument("--default-bz", type=int, default=1024)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_best_rows(root: Path) -> dict[str, dict]:
    rows = []
    for name in ("gpu0_status.jsonl", "gpu1_status.jsonl"):
        rows.extend(read_jsonl(root / name))

    best = {}
    for row in rows:
        task = row.get("task")
        validnll = row.get("validnll")
        if not task or validnll is None:
            continue
        if isinstance(validnll, float) and math.isnan(validnll):
            continue
        cur = best.get(task)
        if cur is None or float(validnll) < float(cur["validnll"]):
            best[task] = row
    return best


def job_done(save_dir: Path) -> bool:
    return (
        (save_dir / "qid_test_question_predictions.txt").exists()
        and (save_dir / "qid_test_question_window_predictions.txt").exists()
    )


def load_config(save_dir: Path) -> dict:
    return json.loads((save_dir / "config.json").read_text(encoding="utf-8"))


def ensure_queue_dirs(queue_dir: Path) -> None:
    for name in ("pending", "inflight", "history/finished", "history/failed"):
        (queue_dir / name).mkdir(parents=True, exist_ok=True)


def build_job(best_row: dict, config: dict, default_bz: int) -> dict:
    params = config["params"]
    train_config = config.get("train_config", {})
    save_dir = str(Path(best_row["model_save_path"]).parent)
    dataset_name = params["dataset_name"]
    model_name = params["model_name"]
    return {
        "job_id": best_row["task"],
        "dataset_name": dataset_name,
        "model_name": model_name,
        "save_dir": save_dir,
        "bz": max(int(default_bz), int(train_config.get("batch_size", 64))),
        "fusion_type": "early_fusion,late_fusion",
        "use_wandb": 0,
        "scenario": "standard",
        "train_ratio": 1.0,
        "question_holdout_ratio": float(params.get("question_holdout_ratio", 0.25)),
        "test_file_override": params.get("test_file_override", ""),
        "test_window_file_override": params.get("test_window_file_override", ""),
        "test_question_file_override": params.get("test_question_file_override", ""),
        "test_question_window_file_override": params.get("test_question_window_file_override", ""),
        "resident_key": resident_family_key(dataset_name, model_name),
        "selection_metric": "validnll",
        "best_validnll": float(best_row["validnll"]),
        "best_validauc": float(best_row.get("validauc", -1)),
        "best_validacc": float(best_row.get("validacc", -1)),
        "best_validbrier": float(best_row.get("validbrier", -1)),
        "best_validece": float(best_row.get("validece", -1)),
        "best_epoch": int(best_row.get("best_epoch", -1)),
        "trial_index": int(best_row.get("trial_index", -1)),
    }


def main() -> int:
    args = parse_args()
    best = load_best_rows(args.root)
    ensure_queue_dirs(args.queue_dir)

    summary_rows = []
    queued = 0
    skipped_done = 0

    for task, best_row in sorted(best.items()):
        save_dir = Path(best_row["model_save_path"]).parent
        config = load_config(save_dir)
        params = config["params"]
        train_config = config.get("train_config", {})

        row = {
            "task": task,
            "dataset_name": params.get("dataset_name"),
            "fold": int(params.get("fold", -1)),
            "seed": int(params.get("seed", -1)),
            "save_dir": str(save_dir),
            "model_save_path": best_row.get("model_save_path", ""),
            "trial_index": int(best_row.get("trial_index", -1)),
            "best_epoch": int(best_row.get("best_epoch", -1)),
            "duration_sec": float(best_row.get("duration_sec", -1)),
            "validnll": float(best_row.get("validnll", -1)),
            "validauc": float(best_row.get("validauc", -1)),
            "validacc": float(best_row.get("validacc", -1)),
            "validbrier": float(best_row.get("validbrier", -1)),
            "validece": float(best_row.get("validece", -1)),
            "train_batch_size": int(train_config.get("batch_size", -1)),
            "d_model": params.get("d_model"),
            "n_blocks": params.get("n_blocks"),
            "dropout": params.get("dropout"),
            "learning_rate": params.get("learning_rate"),
            "final_fc_dim": params.get("final_fc_dim"),
            "final_fc_dim2": params.get("final_fc_dim2"),
            "ctw_feat_dim": params.get("ctw_feat_dim"),
            "ctw_delta_scale": params.get("ctw_delta_scale"),
            "done": job_done(save_dir),
        }
        summary_rows.append(row)

        if row["done"]:
            skipped_done += 1
            continue

        job = build_job(best_row, config, args.default_bz)
        pending_path = args.queue_dir / "pending" / f"{task}.json"
        pending_path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        queued += 1

    args.summary_json.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with args.summary_csv.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(summary_rows[0].keys()) if summary_rows else ["task"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(
        json.dumps(
            {
                "root": str(args.root),
                "queue_dir": str(args.queue_dir),
                "summary_json": str(args.summary_json),
                "summary_csv": str(args.summary_csv),
                "best_task_count": len(summary_rows),
                "queued": queued,
                "skipped_done": skipped_done,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

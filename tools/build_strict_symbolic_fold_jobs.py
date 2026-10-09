#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

build_dataset_fold_aligned_symbolic_artifacts = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build strict symbolic fold-aligned artifacts and write per-fold artifact_info.json files."
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--method", choices=["ctw", "contextmix"], required=True)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--item-col", default="questions")
    parser.add_argument("--depth-or-order", type=int, default=6)
    parser.add_argument("--ctw-backend", default="cpp")
    parser.add_argument("--parallel-folds", type=int, default=1, help="How many folds to build concurrently.")
    parser.add_argument(
        "--jobs-per-fold",
        type=int,
        default=0,
        help="If > 0, override PYKT_SYMBOLIC_BUILD_JOBS inside each fold build process.",
    )
    parser.add_argument(
        "--fold-jobs-root",
        type=Path,
        default=REPO_ROOT / "runs" / "symbolic_hybrid_benchmark_parallel" / "fold_jobs",
        help="Root that will contain <dataset>/<method>_fold<k>/<method>/artifacts",
    )
    return parser.parse_args()


def load_data_config(dataset_name: str) -> dict:
    data_config = json.loads((REPO_ROOT / "configs" / "data_config.json").read_text(encoding="utf-8"))
    cfg = dict(data_config[dataset_name])
    cfg["dataset_name"] = dataset_name
    cfg["dpath"] = str(((REPO_ROOT / "configs") / cfg["dpath"]).resolve())
    return cfg


def parse_folds(text: str) -> list[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def build_one_fold(
    dataset_name: str,
    cfg: dict,
    method: str,
    fold: int,
    item_col: str,
    depth_or_order: int,
    ctw_backend: str,
    fold_jobs_root: str,
    jobs_per_fold: int,
) -> dict:
    if jobs_per_fold > 0:
        os.environ["PYKT_SYMBOLIC_BUILD_JOBS"] = str(int(jobs_per_fold))
    output_root = Path(fold_jobs_root) / dataset_name / f"{method}_fold{fold}" / method / "artifacts"
    output_root.mkdir(parents=True, exist_ok=True)
    start = time.time()
    artifact_info = build_dataset_fold_aligned_symbolic_artifacts(
        data_config=cfg,
        output_root=str(output_root),
        method=method,
        item_col=item_col,
        depth_or_order=depth_or_order,
        ctw_backend=ctw_backend,
        folds=[fold],
    )
    artifact_info_path = output_root.parent / "artifact_info.json"
    artifact_info_path.write_text(json.dumps(artifact_info, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "fold": fold,
        "seconds": round(time.time() - start, 2),
        "artifact_info": str(artifact_info_path),
        "output_root": str(output_root),
    }


def main() -> int:
    args = parse_args()
    from pykt.utils.symbolic_benchmark_builder import build_dataset_fold_aligned_symbolic_artifacts as build_artifacts
    global build_dataset_fold_aligned_symbolic_artifacts
    build_dataset_fold_aligned_symbolic_artifacts = build_artifacts
    cfg = load_data_config(args.dataset_name)
    folds = parse_folds(args.folds)
    effective_parallel_folds = int(args.parallel_folds)
    effective_jobs_per_fold = int(args.jobs_per_fold)
    nested_parallelism_disabled = False
    if effective_parallel_folds > 1 and effective_jobs_per_fold > 1:
        effective_jobs_per_fold = 1
        nested_parallelism_disabled = True

    print(
        json.dumps(
            {
                "stage": "start",
                "dataset": args.dataset_name,
                "method": args.method,
                "folds": folds,
                "item_col": args.item_col,
                "depth_or_order": args.depth_or_order,
                "ctw_backend": args.ctw_backend,
                "jobs": os.environ.get("PYKT_SYMBOLIC_BUILD_JOBS", ""),
                "parallel_folds": int(args.parallel_folds),
                "jobs_per_fold": int(args.jobs_per_fold),
                "effective_parallel_folds": effective_parallel_folds,
                "effective_jobs_per_fold": effective_jobs_per_fold,
                "nested_parallelism_disabled": nested_parallelism_disabled,
                "fold_jobs_root": str(args.fold_jobs_root),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    if effective_parallel_folds <= 1:
        for fold in folds:
            output_root = args.fold_jobs_root / args.dataset_name / f"{args.method}_fold{fold}" / args.method / "artifacts"
            print(
                json.dumps(
                    {"stage": "fold_start", "fold": fold, "output_root": str(output_root)},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            result = build_one_fold(
                dataset_name=args.dataset_name,
                cfg=cfg,
                method=args.method,
                fold=fold,
                item_col=args.item_col,
                depth_or_order=args.depth_or_order,
                ctw_backend=args.ctw_backend,
                fold_jobs_root=str(args.fold_jobs_root),
                jobs_per_fold=effective_jobs_per_fold,
            )
            print(json.dumps({"stage": "fold_done", **result}, ensure_ascii=False), flush=True)
    else:
        max_workers = max(1, min(effective_parallel_folds, len(folds)))
        print(
            json.dumps(
                {"stage": "parallel_mode", "parallel_folds": max_workers},
                ensure_ascii=False,
            ),
            flush=True,
        )
        for fold in folds:
            output_root = args.fold_jobs_root / args.dataset_name / f"{args.method}_fold{fold}" / args.method / "artifacts"
            print(
                json.dumps(
                    {"stage": "fold_queued", "fold": fold, "output_root": str(output_root)},
                    ensure_ascii=False,
                ),
                flush=True,
            )
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    build_one_fold,
                    dataset_name=args.dataset_name,
                    cfg=cfg,
                    method=args.method,
                    fold=fold,
                    item_col=args.item_col,
                    depth_or_order=args.depth_or_order,
                    ctw_backend=args.ctw_backend,
                    fold_jobs_root=str(args.fold_jobs_root),
                    jobs_per_fold=effective_jobs_per_fold,
                ): fold
                for fold in folds
            }
            for future in as_completed(future_map):
                result = future.result()
                print(json.dumps({"stage": "fold_done", **result}, ensure_ascii=False), flush=True)

    print(
        json.dumps(
            {"stage": "done", "dataset": args.dataset_name, "method": args.method, "folds": folds},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

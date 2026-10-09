#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_method_suffix = None
build_fold_aligned_test_symbolic_file = None


TEST_KEY_TO_SRC = {
    "test_file": "test_file",
    "test_window_file": "test_window_file",
    "test_question_file": "test_question_file",
    "test_question_window_file": "test_question_window_file",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build only selected symbolic sidecar files for specific folds.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--method", choices=["ctw", "contextmix"], required=True)
    parser.add_argument("--folds", required=True)
    parser.add_argument(
        "--test-keys",
        required=True,
        help="Comma-separated subset of: test_file,test_window_file,test_question_file,test_question_window_file",
    )
    parser.add_argument("--item-col", default="questions")
    parser.add_argument("--depth-or-order", type=int, default=6)
    parser.add_argument("--ctw-backend", default="cpp")
    parser.add_argument("--parallel-jobs", type=int, default=1)
    parser.add_argument(
        "--fold-jobs-root",
        type=Path,
        default=REPO_ROOT / "runs" / "symbolic_hybrid_benchmark_parallel" / "fold_jobs",
    )
    return parser.parse_args()


def load_data_config(dataset_name: str) -> dict:
    data_config = json.loads((REPO_ROOT / "configs" / "data_config.json").read_text(encoding="utf-8"))
    cfg = dict(data_config[dataset_name])
    cfg["dataset_name"] = dataset_name
    cfg["dpath"] = str(((REPO_ROOT / "configs") / cfg["dpath"]).resolve())
    return cfg


def parse_csv_list(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def parse_folds(text: str) -> list[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def build_one(task: dict) -> dict:
    output_csv = build_fold_aligned_test_symbolic_file(
        sequence_csv=task["sequence_csv"],
        support_csv=task["support_csv"],
        output_csv=task["output_csv"],
        heldout_fold=task["fold"],
        method=task["method"],
        item_col=task["item_col"],
        depth_or_order=task["depth_or_order"],
        ctw_backend=task["ctw_backend"],
    )
    return {
        "fold": task["fold"],
        "test_key": task["test_key"],
        "output_csv": output_csv,
    }


def maybe_write_artifact_info(
    *,
    cfg: dict,
    dataset_name: str,
    method: str,
    fold: int,
    depth_or_order: int,
    ctw_backend: str,
    item_col: str,
    fold_jobs_root: Path,
) -> bool:
    dpath = Path(cfg["dpath"])
    artifact_name = _method_suffix(method, depth_or_order)
    fold_root = fold_jobs_root / dataset_name / f"{method}_fold{fold}" / method
    art_dir = fold_root / "artifacts" / f"fold_{fold}"
    train_valid = art_dir / f"train_valid_sequences_{artifact_name}_strict_fold{fold}.csv"
    if not train_valid.exists():
        return False
    test_by_fold = {}
    for key, src_key in TEST_KEY_TO_SRC.items():
        if src_key not in cfg:
            continue
        src_path = dpath / cfg[src_key]
        if not src_path.exists():
            continue
        base_name = src_path.stem
        out_path = art_dir / f"{base_name}_{artifact_name}_support_excluding_fold{fold}.csv"
        if not out_path.exists():
            return False
        test_by_fold[key] = str(out_path)
    artifact_info = {
        "method": method,
        "item_col": item_col,
        "depth_or_order": int(depth_or_order),
        "ctw_backend": ctw_backend if method == "ctw" else None,
        "folds": [fold],
        "train_valid_by_fold": {str(fold): str(train_valid)},
        "test_by_fold": {str(fold): test_by_fold},
    }
    (fold_root / "artifact_info.json").write_text(json.dumps(artifact_info, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def main() -> int:
    args = parse_args()
    from pykt.utils.symbolic_benchmark_builder import (
        _method_suffix as method_suffix,
        build_fold_aligned_test_symbolic_file as build_file,
    )
    global _method_suffix, build_fold_aligned_test_symbolic_file
    _method_suffix = method_suffix
    build_fold_aligned_test_symbolic_file = build_file
    cfg = load_data_config(args.dataset_name)
    folds = parse_folds(args.folds)
    test_keys = parse_csv_list(args.test_keys)
    for key in test_keys:
        if key not in TEST_KEY_TO_SRC:
            raise ValueError(f"Unsupported test key: {key}")

    dpath = Path(cfg["dpath"])
    train_valid_csv = str(dpath / cfg["train_valid_file"])
    artifact_name = _method_suffix(args.method, args.depth_or_order)
    tasks = []
    for fold in folds:
        art_dir = args.fold_jobs_root / args.dataset_name / f"{args.method}_fold{fold}" / args.method / "artifacts" / f"fold_{fold}"
        art_dir.mkdir(parents=True, exist_ok=True)
        for test_key in test_keys:
            src_path = dpath / cfg[TEST_KEY_TO_SRC[test_key]]
            if not src_path.exists():
                continue
            out_path = art_dir / f"{src_path.stem}_{artifact_name}_support_excluding_fold{fold}.csv"
            if out_path.exists():
                continue
            tasks.append(
                {
                    "fold": fold,
                    "test_key": test_key,
                    "sequence_csv": str(src_path),
                    "support_csv": train_valid_csv,
                    "output_csv": str(out_path),
                    "method": args.method,
                    "item_col": args.item_col,
                    "depth_or_order": int(args.depth_or_order),
                    "ctw_backend": args.ctw_backend,
                }
            )

    print(json.dumps({"stage": "start", "task_count": len(tasks), "folds": folds, "test_keys": test_keys}, ensure_ascii=False), flush=True)
    if tasks:
        max_workers = max(1, min(int(args.parallel_jobs), len(tasks), os.cpu_count() or 1))
        if max_workers <= 1:
            for task in tasks:
                print(json.dumps({"stage": "task_start", "fold": task["fold"], "test_key": task["test_key"]}, ensure_ascii=False), flush=True)
                result = build_one(task)
                print(json.dumps({"stage": "task_done", **result}, ensure_ascii=False), flush=True)
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_map = {executor.submit(build_one, task): task for task in tasks}
                for future in as_completed(future_map):
                    result = future.result()
                    print(json.dumps({"stage": "task_done", **result}, ensure_ascii=False), flush=True)

    artifact_status = {}
    for fold in folds:
        ok = maybe_write_artifact_info(
            cfg=cfg,
            dataset_name=args.dataset_name,
            method=args.method,
            fold=fold,
            depth_or_order=int(args.depth_or_order),
            ctw_backend=args.ctw_backend,
            item_col=args.item_col,
            fold_jobs_root=args.fold_jobs_root,
        )
        artifact_status[str(fold)] = ok
    print(json.dumps({"stage": "done", "artifact_info_written": artifact_status}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

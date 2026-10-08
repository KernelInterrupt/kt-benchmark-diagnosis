#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import pandas as pd


def _parse_tokens(value):
    if pd.isna(value):
        return []
    tokens = str(value).split(",")
    if tokens and tokens[-1] == "":
        tokens = tokens[:-1]
    return tokens


def _safe_int(value, default=-1):
    try:
        return int(value)
    except Exception:
        return default


def _num_concepts(token: str) -> int:
    if token in ("", "-1"):
        return 0
    return len([part for part in str(token).split("_") if part and part != "-1"])


def _iter_question_rows(frame: pd.DataFrame, fold: int):
    for _, row in frame.iterrows():
        qidxs = _parse_tokens(row.get("qidxs"))
        orirows = _parse_tokens(row.get("orirow"))
        responses = _parse_tokens(row.get("responses"))
        concepts = _parse_tokens(row.get("concepts"))
        probs = _parse_tokens(row.get("ctw_pseqs"))
        selectmasks = _parse_tokens(row.get("selectmasks"))
        limit = min(len(qidxs), len(orirows), len(responses), len(concepts), len(probs), len(selectmasks))
        for pos in range(limit):
            if _safe_int(selectmasks[pos]) != 1:
                continue
            qidx = _safe_int(qidxs[pos])
            orirow = _safe_int(orirows[pos])
            y_true = _safe_int(responses[pos])
            if qidx < 0 or orirow < 0 or y_true < 0:
                continue
            try:
                prob = float(probs[pos])
            except Exception:
                continue
            if prob < 0:
                continue
            yield {
                "fold": int(fold),
                "qidx": qidx,
                "orirow": orirow,
                "y_true": y_true,
                "num_concepts": _num_concepts(concepts[pos] if pos < len(concepts) else "-1"),
                "concept_preds": prob,
                "late_mean": prob,
                "late_vote": prob,
                "late_all": prob,
            }


def _load_artifact_info(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def export_method(repo_root: Path, dataset_dir: Path, method: str, output_dir: Path, overwrite: bool) -> list[dict]:
    rows_question = []
    rows_window = []
    for fold_dir in sorted(dataset_dir.glob(f"{method}_fold*")):
        method_dir = fold_dir / method
        artifact_info_path = method_dir / "artifact_info.json"
        if not artifact_info_path.exists():
            continue
        artifact_info = _load_artifact_info(artifact_info_path)
        test_map = artifact_info.get("test_by_fold", {})
        for fold_key, files in test_map.items():
            fold = int(fold_key)
            q_path = repo_root / files["test_question_file"]
            qw_path = repo_root / files["test_question_window_file"]
            if q_path.exists():
                rows_question.extend(_iter_question_rows(pd.read_csv(q_path), fold=fold))
            if qw_path.exists():
                rows_window.extend(_iter_question_rows(pd.read_csv(qw_path), fold=fold))

    out_method_dir = output_dir / method
    out_method_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name, rows in [("question_predictions.csv", rows_question), ("question_window_predictions.csv", rows_window)]:
        out_path = out_method_dir / name
        if out_path.exists() and not overwrite:
            outputs.append({"file": str(out_path), "rows": None, "status": "kept_existing"})
            continue
        df = pd.DataFrame(rows).sort_values(["fold", "orirow", "qidx"]).reset_index(drop=True) if rows else pd.DataFrame(
            columns=["fold", "qidx", "orirow", "y_true", "num_concepts", "concept_preds", "late_mean", "late_vote", "late_all"]
        )
        df.to_csv(out_path, index=False)
        outputs.append({"file": str(out_path), "rows": int(len(df)), "status": "written"})
    return outputs


def main() -> int:
    parser.add_argument("--dataset-root", type=Path, required=True, help="e.g. runs/symbolic_hybrid_benchmark_parallel/fold_jobs/nips_task34")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)
    dataset_root = args.dataset_root if args.dataset_root.is_absolute() else repo_root / args.dataset_root
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    methods = [item.strip().lower() for item in args.methods.split(",") if item.strip()]

    summary = {}
    for method in methods:
        summary[method] = export_method(repo_root, dataset_root, method, output_dir, overwrite=args.overwrite)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

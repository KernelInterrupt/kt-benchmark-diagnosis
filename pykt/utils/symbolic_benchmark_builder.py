from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import math
import pandas as pd

from .ctw_estimator import create_ctw_estimator, interaction_context_token, query_context_token
from .contextmix_estimator import create_contextmix_estimator


def _parse_seq(value: Any) -> List[str]:
    if pd.isna(value):
        return []
    tokens = str(value).split(",")
    if len(tokens) > 0 and tokens[-1] == "":
        tokens = tokens[:-1]
    return tokens


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _seq_to_csv(values: Sequence[float]) -> str:
    return ",".join(str(v) for v in values)


def _method_name(method: str) -> str:
    name = str(method).strip().lower()
    if name not in {"ctw", "contextmix"}:
        raise ValueError("method must be one of: ctw, contextmix")
    return name


def _method_suffix(method: str, depth_or_order: int) -> str:
    name = _method_name(method)
    if name == "ctw":
        return f"ctw_depth{int(depth_or_order)}"
    return f"contextmix_order{int(depth_or_order)}"


def _symbolic_column_names() -> Dict[str, str]:
    # Keep the output schema aligned with the existing residual KT pipeline.
    # The file path encodes which symbolic estimator produced the contents.
    return {
        "p": "ctw_pseqs",
        "logit": "ctw_logitseqs",
        "depth": "ctw_depthseqs",
        "total": "ctw_totalseqs",
        "pos": "ctw_posseqs",
        "neg": "ctw_negseqs",
    }


def _create_estimator(method: str, depth_or_order: int, ctw_backend: str):
    name = _method_name(method)
    if name == "ctw":
        return create_ctw_estimator(max_depth=depth_or_order, backend=ctw_backend)
    return create_contextmix_estimator(max_order=depth_or_order, backend="cpp")


def _resolve_build_jobs(task_count: int) -> int:
    raw = os.getenv("PYKT_SYMBOLIC_BUILD_JOBS", "").strip()
    if raw == "":
        return 1
    try:
        value = int(raw)
    except ValueError:
        return 1
    if value <= 1:
        return 1
    cpu_count = os.cpu_count() or 1
    return max(1, min(value, cpu_count, task_count))


def _run_build_task(task: Mapping[str, Any]) -> Tuple[str, int, str]:
    kind = str(task["kind"])
    fold = int(task["fold"])
    output_csv = str(task["output_csv"])
    if kind == "train_valid":
        build_fold_aligned_train_valid_symbolic_file(
            train_valid_csv=str(task["train_valid_csv"]),
            output_csv=output_csv,
            heldout_fold=fold,
            method=str(task["method"]),
            item_col=str(task["item_col"]),
            depth_or_order=int(task["depth_or_order"]),
            ctw_backend=str(task["ctw_backend"]),
        )
    elif kind == "test":
        build_fold_aligned_test_symbolic_file(
            sequence_csv=str(task["sequence_csv"]),
            support_csv=str(task["support_csv"]),
            output_csv=output_csv,
            heldout_fold=fold,
            method=str(task["method"]),
            item_col=str(task["item_col"]),
            depth_or_order=int(task["depth_or_order"]),
            ctw_backend=str(task["ctw_backend"]),
        )
    else:
        raise ValueError(f"Unsupported build task kind: {kind}")
    return kind, fold, output_csv


def _fit_base_estimator_from_df(
    df: pd.DataFrame,
    method: str,
    item_col: str,
    depth_or_order: int,
    ctw_backend: str,
):
    estimator = _create_estimator(method=method, depth_or_order=depth_or_order, ctw_backend=ctw_backend)
    suffix_len = max(0, int(depth_or_order) - 1)
    for _, row in df.iterrows():
        items = _parse_seq(row[item_col])
        responses = _parse_seq(row["responses"])
        seq_len = min(len(items), len(responses))
        suffix = deque(maxlen=suffix_len)
        for pos in range(seq_len):
            item = str(items[pos])
            response = responses[pos]
            if item == "-1" or response == "-1":
                continue
            target = _safe_int(response)
            context = list(suffix)
            context.append(query_context_token(item))
            estimator.update(context, target)
            suffix.append(interaction_context_token(item, target))
    return estimator


def _prepare_local_estimator(base_estimator):
    if hasattr(base_estimator, "checkpoint") and hasattr(base_estimator, "rollback"):
        marker = base_estimator.checkpoint()
        return base_estimator, marker
    return base_estimator.clone(), None


def _restore_local_estimator(local_estimator, marker) -> None:
    if marker is not None:
        local_estimator.rollback(marker)


def _augment_df_with_row_support(
    df: pd.DataFrame,
    method: str,
    item_col: str,
    depth_or_order: int,
    ctw_backend: str,
    support_key_by_index: Mapping[int, Tuple[int, ...]],
    base_estimators: Mapping[Tuple[int, ...], Any],
) -> pd.DataFrame:
    columns = _symbolic_column_names()
    suffix_len = max(0, int(depth_or_order) - 1)
    pvals_all: List[str] = []
    logitvals_all: List[str] = []
    depthvals_all: List[str] = []
    totalvals_all: List[str] = []
    posvals_all: List[str] = []
    negvals_all: List[str] = []

    for row_index, row in df.iterrows():
        items = _parse_seq(row[item_col])
        responses = _parse_seq(row["responses"])
        seq_len = min(len(items), len(responses))
        support_key = support_key_by_index[int(row_index)]
        base_estimator = base_estimators[support_key]
        local_estimator, marker = _prepare_local_estimator(base_estimator)

        pvals: List[float] = []
        logitvals: List[float] = []
        depthvals: List[float] = []
        totalvals: List[float] = []
        posvals: List[float] = []
        negvals: List[float] = []
        suffix = deque(maxlen=suffix_len)
        try:
            for pos in range(seq_len):
                item = str(items[pos])
                response = responses[pos]
                if item == "-1" or response == "-1":
                    pvals.append(-1.0)
                    logitvals.append(-1.0)
                    depthvals.append(-1.0)
                    totalvals.append(-1.0)
                    posvals.append(-1.0)
                    negvals.append(-1.0)
                    continue
                target = _safe_int(response)
                context = list(suffix)
                context.append(query_context_token(item))
                pred = local_estimator.predict(context)
                p = float(pred.p_correct)
                p_clip = max(min(p, 1.0 - 1e-12), 1e-12)
                pvals.append(p)
                logitvals.append(float(math.log(p_clip / (1.0 - p_clip))))
                depthvals.append(float(pred.deepest_match_depth))
                totalvals.append(float(pred.deepest_match_total))
                posvals.append(float(pred.deepest_match_positive))
                negvals.append(float(pred.deepest_match_negative))
                local_estimator.update(context, target)
                suffix.append(interaction_context_token(item, target))
        finally:
            _restore_local_estimator(local_estimator, marker)

        if len(items) > seq_len:
            pad_len = len(items) - seq_len
            pvals.extend([-1.0] * pad_len)
            logitvals.extend([-1.0] * pad_len)
            depthvals.extend([-1.0] * pad_len)
            totalvals.extend([-1.0] * pad_len)
            posvals.extend([-1.0] * pad_len)
            negvals.extend([-1.0] * pad_len)

        pvals_all.append(_seq_to_csv(pvals))
        logitvals_all.append(_seq_to_csv(logitvals))
        depthvals_all.append(_seq_to_csv(depthvals))
        totalvals_all.append(_seq_to_csv(totalvals))
        posvals_all.append(_seq_to_csv(posvals))
        negvals_all.append(_seq_to_csv(negvals))

    out_df = df.copy()
    out_df[columns["p"]] = pvals_all
    out_df[columns["logit"]] = logitvals_all
    out_df[columns["depth"]] = depthvals_all
    out_df[columns["total"]] = totalvals_all
    out_df[columns["pos"]] = posvals_all
    out_df[columns["neg"]] = negvals_all
    return out_df


def build_fold_aligned_train_valid_symbolic_file(
    train_valid_csv: str,
    output_csv: str,
    heldout_fold: int,
    method: str = "ctw",
    item_col: str = "questions",
    depth_or_order: int = 6,
    ctw_backend: str = "cpp",
) -> str:
    if Path(output_csv).exists():
        return output_csv
    method = _method_name(method)
    df = pd.read_csv(train_valid_csv)
    if "fold" not in df.columns:
        raise ValueError("train_valid_csv must contain 'fold'.")
    if item_col not in df.columns or "responses" not in df.columns:
        raise ValueError(f"train_valid_csv must contain '{item_col}' and 'responses'.")

    heldout_fold = int(heldout_fold)
    support_keys: Dict[int, Tuple[int, ...]] = {}
    excluded_fold_sets = set()
    for row_index, row in df.iterrows():
        row_fold = _safe_int(row["fold"])
        if row_fold == heldout_fold:
            excluded = (heldout_fold,)
        else:
            excluded = tuple(sorted({heldout_fold, row_fold}))
        support_keys[int(row_index)] = excluded
        excluded_fold_sets.add(excluded)

    base_estimators = {}
    for excluded in sorted(excluded_fold_sets):
        support_df = df[~df["fold"].isin(list(excluded))].copy()
        base_estimators[excluded] = _fit_base_estimator_from_df(
            support_df,
            method=method,
            item_col=item_col,
            depth_or_order=depth_or_order,
            ctw_backend=ctw_backend,
        )

    out_df = _augment_df_with_row_support(
        df=df,
        method=method,
        item_col=item_col,
        depth_or_order=depth_or_order,
        ctw_backend=ctw_backend,
        support_key_by_index=support_keys,
        base_estimators=base_estimators,
    )
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    return output_csv


def build_fold_aligned_test_symbolic_file(
    sequence_csv: str,
    support_csv: str,
    output_csv: str,
    heldout_fold: int,
    method: str = "ctw",
    item_col: str = "questions",
    depth_or_order: int = 6,
    ctw_backend: str = "cpp",
) -> str:
    if Path(output_csv).exists():
        return output_csv
    method = _method_name(method)
    df = pd.read_csv(sequence_csv)
    support_df = pd.read_csv(support_csv)
    if "fold" not in support_df.columns:
        raise ValueError("support_csv must contain 'fold'.")
    if item_col not in df.columns or "responses" not in df.columns:
        raise ValueError(f"sequence_csv must contain '{item_col}' and 'responses'.")

    heldout_fold = int(heldout_fold)
    filtered_support_df = support_df[support_df["fold"] != heldout_fold].copy()
    base_estimator = _fit_base_estimator_from_df(
        filtered_support_df,
        method=method,
        item_col=item_col,
        depth_or_order=depth_or_order,
        ctw_backend=ctw_backend,
    )
    support_keys = {int(row_index): (heldout_fold,) for row_index in df.index}
    out_df = _augment_df_with_row_support(
        df=df,
        method=method,
        item_col=item_col,
        depth_or_order=depth_or_order,
        ctw_backend=ctw_backend,
        support_key_by_index=support_keys,
        base_estimators={(heldout_fold,): base_estimator},
    )
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    return output_csv


def build_dataset_fold_aligned_symbolic_artifacts(
    data_config: Mapping[str, Any],
    output_root: str,
    method: str = "ctw",
    item_col: str = "questions",
    depth_or_order: int = 6,
    ctw_backend: str = "cpp",
    folds: Optional[Iterable[int]] = None,
) -> Dict[str, Any]:
    method = _method_name(method)
    dpath = str(data_config["dpath"])
    train_valid_csv = str(Path(dpath) / data_config["train_valid_file"])
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    if folds is None:
        folds = data_config["folds"]
    fold_list = [int(fold) for fold in folds]

    artifact_name = _method_suffix(method, depth_or_order)
    train_valid_outputs: Dict[int, str] = {}
    test_outputs: Dict[int, Dict[str, str]] = {}
    test_keys = [
        "test_file",
        "test_window_file",
        "test_question_file",
        "test_question_window_file",
    ]
    build_tasks: List[Dict[str, Any]] = []

    for fold in fold_list:
        fold_dir = output_root_path / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_valid_output = str(fold_dir / f"train_valid_sequences_{artifact_name}_strict_fold{fold}.csv")
        train_valid_outputs[fold] = train_valid_output
        build_tasks.append(
            {
                "kind": "train_valid",
                "fold": fold,
                "train_valid_csv": train_valid_csv,
                "output_csv": train_valid_output,
                "method": method,
                "item_col": item_col,
                "depth_or_order": int(depth_or_order),
                "ctw_backend": ctw_backend,
            }
        )

        test_outputs[fold] = {}
        for key in test_keys:
            if key not in data_config:
                continue
            src_path = Path(dpath) / data_config[key]
            if not src_path.exists():
                continue
            base_name = src_path.stem
            output_csv = str(fold_dir / f"{base_name}_{artifact_name}_support_excluding_fold{fold}.csv")
            test_outputs[fold][key] = output_csv
            build_tasks.append(
                {
                    "kind": "test",
                    "fold": fold,
                    "sequence_csv": str(src_path),
                    "support_csv": train_valid_csv,
                    "output_csv": output_csv,
                    "method": method,
                    "item_col": item_col,
                    "depth_or_order": int(depth_or_order),
                    "ctw_backend": ctw_backend,
                }
            )

    build_jobs = _resolve_build_jobs(len(build_tasks))
    if build_jobs <= 1:
        for task in build_tasks:
            _run_build_task(task)
    else:
        with ProcessPoolExecutor(max_workers=build_jobs) as executor:
            futures = [executor.submit(_run_build_task, task) for task in build_tasks]
            for future in as_completed(futures):
                future.result()

    return {
        "method": method,
        "item_col": item_col,
        "depth_or_order": int(depth_or_order),
        "ctw_backend": ctw_backend if method == "ctw" else None,
        "folds": fold_list,
        "train_valid_by_fold": train_valid_outputs,
        "test_by_fold": test_outputs,
    }

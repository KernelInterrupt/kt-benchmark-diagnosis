from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import json
import math

import numpy as np
import pandas as pd

from .ctw_estimator import (
    CTWPredictionStats,
    create_ctw_estimator,
    interaction_context_token,
    query_context_token,
)
from .liu_estimator import (
    LIUEstimator,
    LIUStepEstimate,
    aggregate_by_item,
    average_bayes_floor,
    average_liu,
    binary_entropy,
    cosine_kernel_from_dict,
    exact_match_kernel,
)


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


def load_embedding_lookup(embedding_path: str) -> Dict[str, np.ndarray]:
    """
    Load qid -> embedding lookup.

    Supported formats:
    - .pt / .pth
      - dict[qid] -> vector
      - 2D tensor/array with row index == qid
    - .npy
      - 2D ndarray with row index == qid
    - .json
      - object mapping qid -> list[float]
    """
    path = Path(embedding_path)
    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {embedding_path}")

    suffix = path.suffix.lower()
    obj: Any
    if suffix in {".pt", ".pth"}:
        import torch

        obj = torch.load(path, map_location="cpu")
    elif suffix == ".npy":
        obj = np.load(path, allow_pickle=True)
    elif suffix == ".json":
        with open(path, "r", encoding="utf8") as fin:
            obj = json.load(fin)
    else:
        raise ValueError(f"Unsupported embedding format: {suffix}")

    lookup: Dict[str, np.ndarray] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            if value is None:
                continue
            arr = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value, dtype=float)
            lookup[str(key)] = np.asarray(arr, dtype=float)
        return lookup

    if hasattr(obj, "detach"):
        obj = obj.detach().cpu().numpy()
    arr = np.asarray(obj, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Non-dict embedding objects must be 2D arrays/tensors.")
    for idx in range(arr.shape[0]):
        lookup[str(idx)] = arr[idx]
    return lookup


@dataclass
class SequenceRunResult:
    per_step_df: pd.DataFrame
    per_item_df: pd.DataFrame
    summary: Dict[str, Any]


def _normalize_fold_set(folds: Optional[Sequence[int]]) -> Optional[Set[int]]:
    if folds is None:
        return None
    return {int(f) for f in folds}


def _estimate_to_row(
    est: LIUStepEstimate,
    row_index: int,
    uid: Any,
    fold: Optional[int],
    compressed_pos: int,
    original_position: int,
    is_selected: bool,
    estimator_name: str,
    ctw_prediction: Optional[CTWPredictionStats] = None,
) -> Dict[str, Any]:
    return {
        "row_index": int(row_index),
        "uid": uid,
        "fold": fold,
        "estimator": estimator_name,
        "query_item": est.query_item,
        "target_response": est.target_response,
        "compressed_position": int(compressed_pos),
        "original_position": int(original_position),
        "history_length": int(compressed_pos),
        "is_selected": bool(is_selected),
        "p_correct": est.p_correct,
        "p_incorrect": est.p_incorrect,
        "liu_plugin": est.liu_plugin,
        "liu_miller_madow": est.liu_miller_madow,
        "bayes_error_plugin": est.bayes_error_plugin,
        "bayes_error_miller_madow": est.bayes_error_miller_madow,
        "total_kernel_weight": est.total_kernel_weight,
        "effective_sample_size": est.effective_sample_size,
        "positive_weight": est.positive_weight,
        "negative_weight": est.negative_weight,
        "occupied_classes": est.occupied_classes,
        "ctw_deepest_match_depth": (
            int(ctw_prediction.deepest_match_depth) if ctw_prediction is not None else math.nan
        ),
        "ctw_deepest_match_total": (
            int(ctw_prediction.deepest_match_total) if ctw_prediction is not None else math.nan
        ),
        "ctw_deepest_match_positive": (
            int(ctw_prediction.deepest_match_positive) if ctw_prediction is not None else math.nan
        ),
        "ctw_deepest_match_negative": (
            int(ctw_prediction.deepest_match_negative) if ctw_prediction is not None else math.nan
        ),
    }


def _ctw_prediction_to_estimate(
    prediction: CTWPredictionStats,
    query_item: str,
    target_response: int,
    step_index: int,
) -> LIUStepEstimate:
    p_correct = float(prediction.p_correct)
    liu_plugin = float(binary_entropy(p_correct))
    bayes_error = float(min(p_correct, 1.0 - p_correct))
    pos = float(prediction.deepest_match_positive)
    neg = float(prediction.deepest_match_negative)
    total = float(prediction.deepest_match_total)
    occupied = int(pos > 0.0) + int(neg > 0.0)
    return LIUStepEstimate(
        step_index=int(step_index),
        query_item=query_item,
        target_response=int(target_response),
        p_correct=p_correct,
        p_incorrect=float(1.0 - p_correct),
        liu_plugin=liu_plugin,
        liu_miller_madow=liu_plugin,
        bayes_error_plugin=bayes_error,
        bayes_error_miller_madow=bayes_error,
        total_kernel_weight=total,
        effective_sample_size=total,
        positive_weight=pos,
        negative_weight=neg,
        occupied_classes=occupied,
    )


def run_liu_on_processed_csv(
    sequence_csv: str,
    decay_lambda: float = 1.0,
    alpha: float = 1.0,
    min_history: int = 1,
    folds: Optional[Sequence[int]] = None,
    score_only_selected: bool = True,
    item_col: Optional[str] = None,
    embedding_path: Optional[str] = None,
    embedding_min_value: float = 0.0,
    estimator: str = "kernel",
    ctw_max_depth: int = 6,
    ctw_backend: str = "python",
) -> SequenceRunResult:
    """
    Run LIU estimation directly on pyKT processed sequence CSV.

    Expected columns:
    - questions or concepts
    - responses
    - optionally selectmasks, uid, fold
    """
    df = pd.read_csv(sequence_csv)
    fold_set = _normalize_fold_set(folds)
    if fold_set is not None and "fold" in df.columns:
        df = df[df["fold"].isin(fold_set)].copy()

    if item_col is None:
        if "questions" in df.columns:
            item_col = "questions"
        elif "concepts" in df.columns:
            item_col = "concepts"
        else:
            raise ValueError("Processed sequence CSV must contain either 'questions' or 'concepts'.")
    if item_col not in df.columns or "responses" not in df.columns:
        raise ValueError(f"Processed sequence CSV must contain '{item_col}' and 'responses' columns.")

    estimator_name = str(estimator).strip().lower()
    if estimator_name not in {"kernel", "ctw"}:
        raise ValueError("estimator must be one of: kernel, ctw.")

    kernel_name = None
    liu_estimator: Optional[LIUEstimator] = None
    if estimator_name == "kernel":
        if embedding_path:
            embedding_lookup = load_embedding_lookup(embedding_path)
            kernel_fn = cosine_kernel_from_dict(embedding_lookup, min_value=embedding_min_value)
            kernel_name = "cosine_embedding"
        else:
            kernel_fn = exact_match_kernel
            kernel_name = "exact_match"
        liu_estimator = LIUEstimator(decay_lambda=decay_lambda, alpha=alpha, kernel_fn=kernel_fn)
    else:
        if embedding_path:
            raise ValueError("CTW estimator does not use embedding_path. Leave it empty.")
        kernel_name = "ctw_symbolic_context_tree"

    per_step_rows: List[Dict[str, Any]] = []
    all_estimates: List[LIUStepEstimate] = []
    total_rows = 0
    used_rows = 0
    total_real_interactions = 0
    total_scored_steps = 0

    for row_index, row in df.iterrows():
        total_rows += 1
        items = _parse_seq(row[item_col])
        responses = _parse_seq(row["responses"])
        selectmasks = _parse_seq(row["selectmasks"]) if "selectmasks" in row else []

        seq_len = min(len(items), len(responses))
        if len(selectmasks) > 0:
            seq_len = min(seq_len, len(selectmasks))
        if seq_len <= min_history:
            continue

        filtered_items: List[str] = []
        filtered_responses: List[int] = []
        filtered_selectmasks: List[int] = []
        filtered_original_positions: List[int] = []

        for pos in range(seq_len):
            item = str(items[pos])
            r = responses[pos]
            if item == "-1" or r == "-1":
                continue
            filtered_items.append(item)
            filtered_responses.append(_safe_int(r))
            if len(selectmasks) > pos:
                filtered_selectmasks.append(_safe_int(selectmasks[pos], default=1))
            else:
                filtered_selectmasks.append(1)
            filtered_original_positions.append(pos)

        if len(filtered_items) <= min_history:
            continue

        used_rows += 1
        total_real_interactions += len(filtered_items)

        uid = row["uid"] if "uid" in row else None
        fold = _safe_int(row["fold"]) if "fold" in row else None

        if estimator_name == "kernel":
            assert liu_estimator is not None
            for compressed_pos in range(min_history, len(filtered_items)):
                is_selected = filtered_selectmasks[compressed_pos] == 1
                if score_only_selected and not is_selected:
                    continue

                est = liu_estimator.estimate_step(
                    history_items=filtered_items[:compressed_pos],
                    history_responses=filtered_responses[:compressed_pos],
                    query_item=filtered_items[compressed_pos],
                    step_index=compressed_pos,
                )
                est.target_response = filtered_responses[compressed_pos]
                all_estimates.append(est)
                total_scored_steps += 1
                per_step_rows.append(
                    _estimate_to_row(
                        est=est,
                        row_index=int(row_index),
                        uid=uid,
                        fold=fold,
                        compressed_pos=int(compressed_pos),
                        original_position=int(filtered_original_positions[compressed_pos]),
                        is_selected=bool(is_selected),
                        estimator_name=estimator_name,
                    )
                )
        else:
            ctw_estimator = create_ctw_estimator(max_depth=ctw_max_depth, backend=ctw_backend)
            history_suffix_tokens = deque(maxlen=max(0, ctw_max_depth - 1))
            for compressed_pos, (item, target_response) in enumerate(zip(filtered_items, filtered_responses)):
                context_tokens = list(history_suffix_tokens)
                context_tokens.append(query_context_token(item))
                is_selected = filtered_selectmasks[compressed_pos] == 1
                if compressed_pos >= min_history and not (score_only_selected and not is_selected):
                    prediction = ctw_estimator.predict(context_tokens)
                    est = _ctw_prediction_to_estimate(
                        prediction=prediction,
                        query_item=item,
                        target_response=int(target_response),
                        step_index=int(compressed_pos),
                    )
                    all_estimates.append(est)
                    total_scored_steps += 1
                    per_step_rows.append(
                        _estimate_to_row(
                            est=est,
                            row_index=int(row_index),
                            uid=uid,
                            fold=fold,
                            compressed_pos=int(compressed_pos),
                            original_position=int(filtered_original_positions[compressed_pos]),
                            is_selected=bool(is_selected),
                            estimator_name=estimator_name,
                            ctw_prediction=prediction,
                        )
                    )
                ctw_estimator.update(context_tokens, int(target_response))
                history_suffix_tokens.append(interaction_context_token(item, int(target_response)))

    per_step_df = pd.DataFrame(per_step_rows)
    per_item_summary = aggregate_by_item(all_estimates, use_miller_madow=True)
    per_item_rows = []
    for item, stats in per_item_summary.items():
        row = {"query_item": item}
        row.update(stats)
        per_item_rows.append(row)
    per_item_df = pd.DataFrame(per_item_rows).sort_values("count", ascending=False) if per_item_rows else pd.DataFrame(
        columns=["query_item", "count", "ru", "item_bayes_floor", "avg_effective_sample_size", "avg_total_kernel_weight"]
    )

    summary = {
        "sequence_csv": sequence_csv,
        "estimator": estimator_name,
        "kernel": kernel_name,
        "item_col": item_col,
        "embedding_path": embedding_path if estimator_name == "kernel" else None,
        "num_rows_total": int(total_rows),
        "num_rows_used": int(used_rows),
        "num_real_interactions": int(total_real_interactions),
        "num_scored_steps": int(total_scored_steps),
        "decay_lambda": float(decay_lambda) if estimator_name == "kernel" else None,
        "alpha": float(alpha) if estimator_name == "kernel" else None,
        "ctw_max_depth": int(ctw_max_depth) if estimator_name == "ctw" else None,
        "ctw_backend": str(ctw_backend) if estimator_name == "ctw" else None,
        "ctw_symbol_mode": "item_response_history_plus_query_token" if estimator_name == "ctw" else None,
        "min_history": int(min_history),
        "score_only_selected": bool(score_only_selected),
        "mean_liu_plugin": float(average_liu(all_estimates, use_miller_madow=False)) if all_estimates else math.nan,
        "mean_liu_miller_madow": float(average_liu(all_estimates, use_miller_madow=True)) if all_estimates else math.nan,
        "mean_bayes_floor_plugin": float(average_bayes_floor(all_estimates, use_miller_madow=False)) if all_estimates else math.nan,
        "mean_bayes_floor_miller_madow": float(average_bayes_floor(all_estimates, use_miller_madow=True)) if all_estimates else math.nan,
        "num_unique_items_scored": int(per_item_df["query_item"].nunique()) if not per_item_df.empty else 0,
    }
    return SequenceRunResult(per_step_df=per_step_df, per_item_df=per_item_df, summary=summary)


def save_sequence_run_result(result: SequenceRunResult, output_dir: str) -> Dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    per_step_path = output_path / "liu_per_step.csv"
    per_item_path = output_path / "liu_per_item.csv"
    summary_path = output_path / "liu_summary.json"

    result.per_step_df.to_csv(per_step_path, index=False)
    result.per_item_df.to_csv(per_item_path, index=False)
    with open(summary_path, "w", encoding="utf8") as fout:
        json.dump(result.summary, fout, ensure_ascii=False, indent=2)

    return {
        "per_step_csv": str(per_step_path),
        "per_item_csv": str(per_item_path),
        "summary_json": str(summary_path),
    }

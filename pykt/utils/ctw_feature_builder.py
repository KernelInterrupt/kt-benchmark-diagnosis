from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional

import math
import pandas as pd

from .ctw_estimator import create_ctw_estimator, interaction_context_token, query_context_token


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


def _seq_to_csv(values: List[float]) -> str:
    return ",".join(str(v) for v in values)


def _fit_base_ctw_from_df(df: pd.DataFrame, item_col: str, max_depth: int, backend: str):
    estimator = create_ctw_estimator(max_depth=max_depth, backend=backend)
    for _, row in df.iterrows():
        items = _parse_seq(row[item_col])
        responses = _parse_seq(row["responses"])
        seq_len = min(len(items), len(responses))
        suffix = deque(maxlen=max(0, max_depth - 1))
        for pos in range(seq_len):
            item = str(items[pos])
            response = responses[pos]
            if item == "-1" or response == "-1":
                continue
            context = list(suffix)
            context.append(query_context_token(item))
            estimator.update(context, _safe_int(response))
            suffix.append(interaction_context_token(item, _safe_int(response)))
    return estimator


def augment_sequence_csv_with_ctw(
    sequence_csv: str,
    output_csv: str,
    item_col: str = "questions",
    support_csv: Optional[str] = None,
    max_depth: int = 6,
    backend: str = "python",
) -> str:
    df = pd.read_csv(sequence_csv)
    if item_col not in df.columns or "responses" not in df.columns:
        raise ValueError(f"{sequence_csv} must contain '{item_col}' and 'responses'.")

    if support_csv:
        support_df = pd.read_csv(support_csv)
        shared_base = _fit_base_ctw_from_df(support_df, item_col=item_col, max_depth=max_depth, backend=backend)
        base_by_fold = None
    else:
        if "fold" not in df.columns:
            raise ValueError("sequence_csv must contain 'fold' when support_csv is not provided.")
        base_by_fold = {}
        for fold in sorted(df["fold"].dropna().astype(int).unique().tolist()):
            train_df = df[df["fold"] != int(fold)].copy()
            base_by_fold[int(fold)] = _fit_base_ctw_from_df(
                train_df, item_col=item_col, max_depth=max_depth, backend=backend
            )
        shared_base = None

    ctw_pseqs = []
    ctw_logitseqs = []
    ctw_depthseqs = []
    ctw_totalseqs = []
    ctw_posseqs = []
    ctw_negseqs = []

    for _, row in df.iterrows():
        items = _parse_seq(row[item_col])
        responses = _parse_seq(row["responses"])
        seq_len = min(len(items), len(responses))
        fold = _safe_int(row["fold"]) if "fold" in row else -1
        base_est = shared_base if shared_base is not None else base_by_fold[int(fold)]
        if backend == "cpp" and hasattr(base_est, "checkpoint") and hasattr(base_est, "rollback"):
            local_est = base_est
            marker = local_est.checkpoint()
        else:
            local_est = base_est.clone()
            marker = None

        pvals: List[float] = []
        logitvals: List[float] = []
        depthvals: List[float] = []
        totalvals: List[float] = []
        posvals: List[float] = []
        negvals: List[float] = []
        suffix = deque(maxlen=max(0, max_depth - 1))
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
                context = list(suffix)
                context.append(query_context_token(item))
                pred = local_est.predict(context)
                p = float(pred.p_correct)
                pvals.append(p)
                logitvals.append(float(math.log(max(min(p, 1 - 1e-12), 1e-12) / max(min(1 - p, 1 - 1e-12), 1e-12))))
                depthvals.append(float(pred.deepest_match_depth))
                totalvals.append(float(pred.deepest_match_total))
                posvals.append(float(pred.deepest_match_positive))
                negvals.append(float(pred.deepest_match_negative))
                local_est.update(context, _safe_int(response))
                suffix.append(interaction_context_token(item, _safe_int(response)))
        finally:
            if marker is not None:
                local_est.rollback(marker)

        if len(items) > seq_len:
            pad_len = len(items) - seq_len
            pvals.extend([-1.0] * pad_len)
            logitvals.extend([-1.0] * pad_len)
            depthvals.extend([-1.0] * pad_len)
            totalvals.extend([-1.0] * pad_len)
            posvals.extend([-1.0] * pad_len)
            negvals.extend([-1.0] * pad_len)

        ctw_pseqs.append(_seq_to_csv(pvals))
        ctw_logitseqs.append(_seq_to_csv(logitvals))
        ctw_depthseqs.append(_seq_to_csv(depthvals))
        ctw_totalseqs.append(_seq_to_csv(totalvals))
        ctw_posseqs.append(_seq_to_csv(posvals))
        ctw_negseqs.append(_seq_to_csv(negvals))

    out_df = df.copy()
    out_df["ctw_pseqs"] = ctw_pseqs
    out_df["ctw_logitseqs"] = ctw_logitseqs
    out_df["ctw_depthseqs"] = ctw_depthseqs
    out_df["ctw_totalseqs"] = ctw_totalseqs
    out_df["ctw_posseqs"] = ctw_posseqs
    out_df["ctw_negseqs"] = ctw_negseqs
    out_df.to_csv(output_csv, index=False)
    return output_csv

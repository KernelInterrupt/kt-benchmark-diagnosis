from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import json
import math

import numpy as np
import pandas as pd
from sklearn import metrics

from .ctw_estimator import (
    create_ctw_estimator,
    interaction_context_token,
    query_context_token,
)
from .liu_estimator import binary_entropy
from .liu_sequence_runner import load_embedding_lookup


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


def _late_fusion_scores(preds: Sequence[float]) -> Dict[str, float]:
    preds = [float(p) for p in preds]
    high = [p for p in preds if p >= 0.5]
    low = [p for p in preds if p < 0.5]
    correctnum = len(high)
    if len(preds) == 0:
        return {"late_mean": math.nan, "late_vote": math.nan, "late_all": math.nan}
    late_mean = float(np.mean(preds))
    if correctnum / len(preds) >= 0.5:
        late_vote = float(np.mean(high)) if len(high) > 0 else 0.5
    else:
        late_vote = float(np.mean(low)) if len(low) > 0 else 0.5
    if correctnum == len(preds):
        late_all = float(np.mean(high)) if len(high) > 0 else 0.5
    else:
        late_all = float(np.mean(low)) if len(low) > 0 else 0.5
    return {"late_mean": late_mean, "late_vote": late_vote, "late_all": late_all}


def _classification_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0:
        return {"auc": math.nan, "acc": math.nan, "nll": math.nan}
    y_pred = (y_score >= 0.5).astype(int)
    eps = 1e-12
    y_score_clip = np.clip(y_score, eps, 1.0 - eps)
    return {
        "auc": float(metrics.roc_auc_score(y_true=y_true, y_score=y_score)),
        "acc": float(metrics.accuracy_score(y_true, y_pred)),
        "nll": float(-(y_true * np.log(y_score_clip) + (1 - y_true) * np.log(1 - y_score_clip)).mean()),
    }


@dataclass
class BenchmarkCeilingResult:
    fold_metrics: pd.DataFrame
    question_predictions: pd.DataFrame
    concept_predictions: pd.DataFrame
    summary: Dict[str, Any]


def _build_valid_sequence_fields(
    row: pd.Series,
    item_col: str,
    require_question_meta: bool,
) -> Dict[str, List[Any]]:
    items = _parse_seq(row[item_col])
    responses = _parse_seq(row["responses"])
    selectmasks = _parse_seq(row["selectmasks"]) if "selectmasks" in row else []
    qidxs = _parse_seq(row["qidxs"]) if "qidxs" in row else []
    orirows = _parse_seq(row["orirow"]) if "orirow" in row else []

    seq_len = min(len(items), len(responses))
    if len(selectmasks) > 0:
        seq_len = min(seq_len, len(selectmasks))
    if len(qidxs) > 0:
        seq_len = min(seq_len, len(qidxs))
    if len(orirows) > 0:
        seq_len = min(seq_len, len(orirows))
    if seq_len == 0:
        return {
            "items": [],
            "responses": [],
            "selects": [],
            "qidxs": [],
            "orirows": [],
        }

    valid_items: List[str] = []
    valid_responses: List[int] = []
    valid_selects: List[int] = []
    valid_qidxs: List[int] = []
    valid_orirows: List[int] = []

    for pos in range(seq_len):
        item = str(items[pos])
        response = responses[pos]
        if item == "-1" or response == "-1":
            continue
        valid_items.append(item)
        valid_responses.append(_safe_int(response))
        valid_selects.append(_safe_int(selectmasks[pos], default=1) if len(selectmasks) > 0 else 1)
        if require_question_meta:
            valid_qidxs.append(_safe_int(qidxs[pos]))
            valid_orirows.append(_safe_int(orirows[pos]))

    return {
        "items": valid_items,
        "responses": valid_responses,
        "selects": valid_selects,
        "qidxs": valid_qidxs,
        "orirows": valid_orirows,
    }


class SupportBank:
    def __init__(
        self,
        support_csv: str,
        item_col: str,
        folds: Sequence[int],
        embedding_path: Optional[str] = None,
        embedding_min_value: float = 0.0,
    ) -> None:
        self.support_csv = support_csv
        self.item_col = item_col
        self.folds = [int(f) for f in folds]
        self.embedding_path = embedding_path
        self.embedding_min_value = embedding_min_value
        self.global_counts: Dict[str, float] = {}
        self.global_pos: Dict[str, float] = {}
        self.fold_counts: Dict[int, Dict[str, float]] = {f: {} for f in self.folds}
        self.fold_pos: Dict[int, Dict[str, float]] = {f: {} for f in self.folds}
        self.total_interactions = 0
        self._build()
        self.items = sorted(self.global_counts.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
        self.item_to_index = {item: idx for idx, item in enumerate(self.items)}
        self._build_vectors()

    def _build(self) -> None:
        df = pd.read_csv(self.support_csv)
        if "fold" not in df.columns:
            raise ValueError("support_csv must contain 'fold'.")
        if self.item_col not in df.columns or "responses" not in df.columns:
            raise ValueError(f"support_csv must contain '{self.item_col}' and 'responses'.")

        for _, row in df.iterrows():
            fold = _safe_int(row["fold"])
            items = _parse_seq(row[self.item_col])
            responses = _parse_seq(row["responses"])
            seq_len = min(len(items), len(responses))
            for pos in range(seq_len):
                item = str(items[pos])
                response = responses[pos]
                if item == "-1" or response == "-1":
                    continue
                r = _safe_int(response)
                self.total_interactions += 1
                self.global_counts[item] = self.global_counts.get(item, 0.0) + 1.0
                self.global_pos[item] = self.global_pos.get(item, 0.0) + float(r)
                self.fold_counts.setdefault(fold, {})
                self.fold_pos.setdefault(fold, {})
                self.fold_counts[fold][item] = self.fold_counts[fold].get(item, 0.0) + 1.0
                self.fold_pos[fold][item] = self.fold_pos[fold].get(item, 0.0) + float(r)

    def _build_vectors(self) -> None:
        n_items = len(self.items)
        self.global_count_vec = np.zeros(n_items, dtype=float)
        self.global_pos_vec = np.zeros(n_items, dtype=float)
        self.fold_count_vecs: Dict[int, np.ndarray] = {}
        self.fold_pos_vecs: Dict[int, np.ndarray] = {}

        for item, idx in self.item_to_index.items():
            self.global_count_vec[idx] = self.global_counts[item]
            self.global_pos_vec[idx] = self.global_pos[item]

        for fold in self.folds:
            cvec = np.zeros(n_items, dtype=float)
            pvec = np.zeros(n_items, dtype=float)
            for item, val in self.fold_counts.get(fold, {}).items():
                cvec[self.item_to_index[item]] = val
            for item, val in self.fold_pos.get(fold, {}).items():
                pvec[self.item_to_index[item]] = val
            self.fold_count_vecs[fold] = cvec
            self.fold_pos_vecs[fold] = pvec

        self.support_count_vecs: Dict[int, np.ndarray] = {
            fold: self.global_count_vec - self.fold_count_vecs[fold] for fold in self.folds
        }
        self.support_pos_vecs: Dict[int, np.ndarray] = {
            fold: self.global_pos_vec - self.fold_pos_vecs[fold] for fold in self.folds
        }

        if self.embedding_path:
            emb_lookup = load_embedding_lookup(self.embedding_path)
            dim = None
            item_embs = []
            for item in self.items:
                vec = emb_lookup.get(str(item))
                if vec is None:
                    if dim is None:
                        dim = len(next(iter(emb_lookup.values())))
                    item_embs.append(np.zeros(dim, dtype=float))
                else:
                    arr = np.asarray(vec, dtype=float)
                    if dim is None:
                        dim = arr.shape[0]
                    item_embs.append(arr)
            mat = np.vstack(item_embs)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms <= 0] = 1.0
            self.norm_mat = mat / norms
            sim = self.norm_mat @ self.norm_mat.T
            sim = np.maximum(sim, self.embedding_min_value)
            self.sim_matrix = sim
            self.kernel_name = "cosine_embedding"
        else:
            self.norm_mat = None
            self.sim_matrix = np.eye(n_items, dtype=float)
            self.kernel_name = "exact_match"

    def kernel_row_for_item(self, item: str) -> np.ndarray:
        idx = self.item_to_index.get(str(item))
        if idx is None:
            return np.zeros(len(self.items), dtype=float)
        return self.sim_matrix[idx]

    def support_contribution(self, fold: int, item: str) -> Tuple[float, float]:
        kernel_row = self.kernel_row_for_item(item)
        total = float(kernel_row @ self.support_count_vecs[fold])
        pos = float(kernel_row @ self.support_pos_vecs[fold])
        return total, pos


def _history_contribution(
    history_items: Sequence[str],
    history_responses: Sequence[int],
    query_item: str,
    bank: SupportBank,
    decay_lambda: float,
) -> Tuple[float, float]:
    if len(history_items) == 0:
        return 0.0, 0.0
    exponents = np.arange(len(history_items) - 1, -1, -1, dtype=float)
    temporal = np.power(decay_lambda, exponents)
    temporal = temporal / temporal.sum()
    hitems = np.array(history_items, dtype=object)
    hresp = np.array(history_responses, dtype=float)
    qidx = bank.item_to_index.get(str(query_item))
    if qidx is None:
        return 0.0, 0.0
    sims = np.array(
        [
            bank.sim_matrix[qidx, bank.item_to_index.get(str(item), qidx)] if str(item) in bank.item_to_index else 0.0
            for item in hitems
        ],
        dtype=float,
    )
    weights = temporal * sims
    total = float(weights.sum())
    pos = float(np.dot(weights, hresp))
    return total, pos


def _fit_ctw_on_support_sequences(
    support_csv: str,
    item_col: str,
    heldout_fold: int,
    max_depth: int,
    backend: str,
):
    df = pd.read_csv(support_csv)
    if "fold" not in df.columns:
        raise ValueError("support_csv must contain 'fold'.")
    if item_col not in df.columns or "responses" not in df.columns:
        raise ValueError(f"support_csv must contain '{item_col}' and 'responses'.")

    estimator = create_ctw_estimator(max_depth=max_depth, backend=backend)
    interactions = 0
    for _, row in df.iterrows():
        if _safe_int(row["fold"]) == int(heldout_fold):
            continue
        fields = _build_valid_sequence_fields(row, item_col=item_col, require_question_meta=False)
        history_suffix_tokens = deque(maxlen=max(0, max_depth - 1))
        for item, response in zip(fields["items"], fields["responses"]):
            context_tokens = list(history_suffix_tokens)
            context_tokens.append(query_context_token(item))
            estimator.update(context_tokens, int(response))
            history_suffix_tokens.append(interaction_context_token(item, int(response)))
            interactions += 1
    return estimator, interactions


def run_folded_question_ceiling(
    support_csv: str,
    eval_question_csv: str,
    output_dir: str,
    folds: Sequence[int],
    item_col: str = "questions",
    decay_lambda: float = 0.9,
    alpha: float = 1.0,
    embedding_path: Optional[str] = None,
    embedding_min_value: float = 0.0,
    history_weight: float = 1.0,
    estimator: str = "kernel",
    ctw_max_depth: int = 6,
    ctw_backend: str = "python",
) -> BenchmarkCeilingResult:
    estimator_name = str(estimator).strip().lower()
    if estimator_name not in {"kernel", "ctw"}:
        raise ValueError("estimator must be one of: kernel, ctw.")

    if estimator_name == "kernel":
        bank = SupportBank(
            support_csv=support_csv,
            item_col=item_col,
            folds=folds,
            embedding_path=embedding_path,
            embedding_min_value=embedding_min_value,
        )
        support_kernel_name = bank.kernel_name
        num_support_interactions = int(bank.total_interactions)
    else:
        if embedding_path:
            raise ValueError("CTW benchmark path does not use embedding_path. Leave it empty.")
        bank = None
        support_kernel_name = "ctw_symbolic_context_tree"
        num_support_interactions = 0
    eval_df = pd.read_csv(eval_question_csv)
    required = {"responses", "selectmasks", "qidxs", "orirow"}
    if item_col not in eval_df.columns:
        raise ValueError(f"eval_question_csv must contain '{item_col}'.")
    missing = [c for c in required if c not in eval_df.columns]
    if missing:
        raise ValueError(f"eval_question_csv missing columns: {missing}")

    concept_rows: List[Dict[str, Any]] = []
    question_rows: List[Dict[str, Any]] = []
    fold_metrics: List[Dict[str, Any]] = []

    support_interactions_per_fold: List[int] = []
    for fold in folds:
        fold_concepts: List[Dict[str, Any]] = []
        if estimator_name == "ctw":
            base_ctw, support_interactions_for_fold = _fit_ctw_on_support_sequences(
                support_csv=support_csv,
                item_col=item_col,
                heldout_fold=int(fold),
                max_depth=ctw_max_depth,
                backend=ctw_backend,
            )
            support_interactions_per_fold.append(int(support_interactions_for_fold))
        else:
            support_interactions_per_fold.append(num_support_interactions)
        for row_idx, row in eval_df.iterrows():
            fields = _build_valid_sequence_fields(row, item_col=item_col, require_question_meta=True)
            valid_items = fields["items"]
            valid_responses = fields["responses"]
            valid_qidxs = fields["qidxs"]
            valid_orirows = fields["orirows"]
            valid_selects = fields["selects"]
            if len(valid_items) == 0:
                continue

            if estimator_name == "kernel":
                assert bank is not None
                for pos in range(len(valid_items)):
                    if valid_selects[pos] != 1:
                        continue
                    query_item = valid_items[pos]
                    target = valid_responses[pos]
                    history_items = valid_items[:pos]
                    history_responses = valid_responses[:pos]
                    support_total, support_pos = bank.support_contribution(fold, query_item)
                    hist_total, hist_pos = _history_contribution(history_items, history_responses, query_item, bank, decay_lambda)
                    total = support_total + history_weight * hist_total
                    pos_weight = support_pos + history_weight * hist_pos
                    p_correct = (pos_weight + alpha) / (total + 2.0 * alpha)
                    liu = binary_entropy(p_correct)
                    bayes = min(p_correct, 1.0 - p_correct)
                    record = {
                        "fold": int(fold),
                        "row_index": int(row_idx),
                        "orirow": int(valid_orirows[pos]),
                        "qidx": int(valid_qidxs[pos]),
                        "query_item": query_item,
                        "target_response": int(target),
                        "history_length": int(pos),
                        "support_total_weight": float(support_total),
                        "support_positive_weight": float(support_pos),
                        "history_total_weight": float(hist_total),
                        "history_positive_weight": float(hist_pos),
                        "p_correct": float(p_correct),
                        "liu_plugin": float(liu),
                        "bayes_error_plugin": float(bayes),
                        "estimator": estimator_name,
                        "ctw_deepest_match_depth": math.nan,
                        "ctw_deepest_match_total": math.nan,
                        "ctw_deepest_match_positive": math.nan,
                        "ctw_deepest_match_negative": math.nan,
                    }
                    fold_concepts.append(record)
                    concept_rows.append(record)
            else:
                if ctw_backend == "cpp" and hasattr(base_ctw, "checkpoint") and hasattr(base_ctw, "rollback"):
                    local_ctw = base_ctw
                    marker = local_ctw.checkpoint()
                    try:
                        history_suffix_tokens = deque(maxlen=max(0, ctw_max_depth - 1))
                        for pos, (query_item, target) in enumerate(zip(valid_items, valid_responses)):
                            context_tokens = list(history_suffix_tokens)
                            context_tokens.append(query_context_token(query_item))
                            if valid_selects[pos] == 1:
                                prediction = local_ctw.predict(context_tokens)
                                p_correct = float(prediction.p_correct)
                                liu = binary_entropy(p_correct)
                                bayes = min(p_correct, 1.0 - p_correct)
                                record = {
                                    "fold": int(fold),
                                    "row_index": int(row_idx),
                                    "orirow": int(valid_orirows[pos]),
                                    "qidx": int(valid_qidxs[pos]),
                                    "query_item": query_item,
                                    "target_response": int(target),
                                    "history_length": int(pos),
                                    "support_total_weight": math.nan,
                                    "support_positive_weight": math.nan,
                                    "history_total_weight": math.nan,
                                    "history_positive_weight": math.nan,
                                    "p_correct": p_correct,
                                    "liu_plugin": float(liu),
                                    "bayes_error_plugin": float(bayes),
                                    "estimator": estimator_name,
                                    "ctw_deepest_match_depth": int(prediction.deepest_match_depth),
                                    "ctw_deepest_match_total": int(prediction.deepest_match_total),
                                    "ctw_deepest_match_positive": int(prediction.deepest_match_positive),
                                    "ctw_deepest_match_negative": int(prediction.deepest_match_negative),
                                }
                                fold_concepts.append(record)
                                concept_rows.append(record)
                            local_ctw.update(context_tokens, int(target))
                            history_suffix_tokens.append(interaction_context_token(query_item, int(target)))
                    finally:
                        local_ctw.rollback(marker)
                else:
                    local_ctw = base_ctw.clone()
                    history_suffix_tokens = deque(maxlen=max(0, ctw_max_depth - 1))
                    for pos, (query_item, target) in enumerate(zip(valid_items, valid_responses)):
                        context_tokens = list(history_suffix_tokens)
                        context_tokens.append(query_context_token(query_item))
                        if valid_selects[pos] == 1:
                            prediction = local_ctw.predict(context_tokens)
                            p_correct = float(prediction.p_correct)
                            liu = binary_entropy(p_correct)
                            bayes = min(p_correct, 1.0 - p_correct)
                            record = {
                                "fold": int(fold),
                                "row_index": int(row_idx),
                                "orirow": int(valid_orirows[pos]),
                                "qidx": int(valid_qidxs[pos]),
                                "query_item": query_item,
                                "target_response": int(target),
                                "history_length": int(pos),
                                "support_total_weight": math.nan,
                                "support_positive_weight": math.nan,
                                "history_total_weight": math.nan,
                                "history_positive_weight": math.nan,
                                "p_correct": p_correct,
                                "liu_plugin": float(liu),
                                "bayes_error_plugin": float(bayes),
                                "estimator": estimator_name,
                                "ctw_deepest_match_depth": int(prediction.deepest_match_depth),
                                "ctw_deepest_match_total": int(prediction.deepest_match_total),
                                "ctw_deepest_match_positive": int(prediction.deepest_match_positive),
                                "ctw_deepest_match_negative": int(prediction.deepest_match_negative),
                            }
                            fold_concepts.append(record)
                            concept_rows.append(record)
                        local_ctw.update(context_tokens, int(target))
                        history_suffix_tokens.append(interaction_context_token(query_item, int(target)))

        concept_df = pd.DataFrame(fold_concepts)
        concept_metrics = _classification_metrics(
            concept_df["target_response"].to_numpy(dtype=int),
            concept_df["p_correct"].to_numpy(dtype=float),
        )
        fold_record = {
            "fold": int(fold),
            "concept_auc": concept_metrics["auc"],
            "concept_acc": concept_metrics["acc"],
            "concept_nll": concept_metrics["nll"],
            "num_concept_points": int(len(concept_df)),
        }

        qrows_for_fold = []
        for qidx, group in concept_df.groupby("qidx", sort=True):
            preds = group["p_correct"].tolist()
            y_true = int(round(float(group["target_response"].mean())))
            fused = _late_fusion_scores(preds)
            qrow = {
                "fold": int(fold),
                "qidx": int(qidx),
                "orirow": int(group["orirow"].iloc[0]),
                "y_true": y_true,
                "num_concepts": int(len(group)),
                "concept_preds": ",".join([str(round(float(x), 6)) for x in preds]),
            }
            qrow.update(fused)
            qrows_for_fold.append(qrow)
            question_rows.append(qrow)
        qdf = pd.DataFrame(qrows_for_fold)
        for key in ["late_mean", "late_vote", "late_all"]:
            m = _classification_metrics(qdf["y_true"].to_numpy(dtype=int), qdf[key].to_numpy(dtype=float))
            fold_record[f"{key}_auc"] = m["auc"]
            fold_record[f"{key}_acc"] = m["acc"]
            fold_record[f"{key}_nll"] = m["nll"]
        fold_record["num_questions"] = int(len(qdf))
        fold_metrics.append(fold_record)

    fold_metrics_df = pd.DataFrame(fold_metrics)
    concept_pred_df = pd.DataFrame(concept_rows)
    question_pred_df = pd.DataFrame(question_rows)

    summary_num_support_interactions = (
        int(num_support_interactions)
        if estimator_name == "kernel"
        else int(round(float(np.mean(support_interactions_per_fold)))) if len(support_interactions_per_fold) > 0 else 0
    )
    summary: Dict[str, Any] = {
        "support_csv": support_csv,
        "eval_question_csv": eval_question_csv,
        "estimator": estimator_name,
        "kernel": support_kernel_name,
        "item_col": item_col,
        "embedding_path": embedding_path if estimator_name == "kernel" else None,
        "folds": [int(f) for f in folds],
        "decay_lambda": float(decay_lambda) if estimator_name == "kernel" else None,
        "alpha": float(alpha) if estimator_name == "kernel" else None,
        "history_weight": float(history_weight) if estimator_name == "kernel" else None,
        "ctw_max_depth": int(ctw_max_depth) if estimator_name == "ctw" else None,
        "ctw_backend": str(ctw_backend) if estimator_name == "ctw" else None,
        "ctw_symbol_mode": "item_response_history_plus_query_token" if estimator_name == "ctw" else None,
        "num_support_interactions": summary_num_support_interactions,
        "support_interactions_per_fold": support_interactions_per_fold,
        "num_eval_concept_points_total": int(len(concept_pred_df)),
        "num_eval_questions_total": int(len(question_pred_df)),
    }
    for metric_col in [c for c in fold_metrics_df.columns if c != "fold"]:
        summary[f"{metric_col}_mean"] = float(fold_metrics_df[metric_col].mean())
        summary[f"{metric_col}_std"] = float(fold_metrics_df[metric_col].std(ddof=0))

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    fold_metrics_df.to_csv(outdir / "fold_metrics.csv", index=False)
    concept_pred_df.to_csv(outdir / "concept_predictions.csv", index=False)
    question_pred_df.to_csv(outdir / "question_predictions.csv", index=False)
    with open(outdir / "summary.json", "w", encoding="utf8") as fout:
        json.dump(summary, fout, ensure_ascii=False, indent=2)

    return BenchmarkCeilingResult(
        fold_metrics=fold_metrics_df,
        question_predictions=question_pred_df,
        concept_predictions=concept_pred_df,
        summary=summary,
    )

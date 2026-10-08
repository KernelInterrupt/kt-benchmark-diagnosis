from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import json
import math

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import HistGradientBoostingClassifier

from .liu_benchmark_runner import _classification_metrics, _late_fusion_scores
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


def _safe_float_div(num: float, den: float, default: float = 0.5) -> float:
    return float(num / den) if den > 0 else float(default)


def _clip_log1p(x: float) -> float:
    return float(math.log1p(max(0.0, x)))


class FeatureSupportBank:
    def __init__(
        self,
        sequence_csv: str,
        folds: Sequence[int],
        question_embedding_path: Optional[str] = None,
        embedding_min_value: float = 0.0,
    ) -> None:
        self.sequence_csv = sequence_csv
        self.folds = [int(f) for f in folds]
        self.question_embedding_path = question_embedding_path
        self.embedding_min_value = embedding_min_value

        self.global_q_count: Dict[str, float] = {}
        self.global_q_pos: Dict[str, float] = {}
        self.global_c_count: Dict[str, float] = {}
        self.global_c_pos: Dict[str, float] = {}
        self.fold_q_count: Dict[int, Dict[str, float]] = {}
        self.fold_q_pos: Dict[int, Dict[str, float]] = {}
        self.fold_c_count: Dict[int, Dict[str, float]] = {}
        self.fold_c_pos: Dict[int, Dict[str, float]] = {}
        self._build_counts()
        self._build_similarity()

    def _build_counts(self) -> None:
        df = pd.read_csv(self.sequence_csv)
        for _, row in df.iterrows():
            fold = _safe_int(row["fold"])
            questions = _parse_seq(row["questions"]) if "questions" in row else []
            concepts = _parse_seq(row["concepts"]) if "concepts" in row else []
            responses = _parse_seq(row["responses"])
            seq_len = min(len(responses), len(questions) if questions else len(responses), len(concepts) if concepts else len(responses))
            for pos in range(seq_len):
                q = str(questions[pos]) if questions else None
                c = str(concepts[pos]) if concepts else None
                r = responses[pos]
                if r == "-1":
                    continue
                y = float(_safe_int(r))
                if q is not None and q != "-1":
                    self.global_q_count[q] = self.global_q_count.get(q, 0.0) + 1.0
                    self.global_q_pos[q] = self.global_q_pos.get(q, 0.0) + y
                    self.fold_q_count.setdefault(fold, {})
                    self.fold_q_pos.setdefault(fold, {})
                    self.fold_q_count[fold][q] = self.fold_q_count[fold].get(q, 0.0) + 1.0
                    self.fold_q_pos[fold][q] = self.fold_q_pos[fold].get(q, 0.0) + y
                if c is not None and c != "-1":
                    self.global_c_count[c] = self.global_c_count.get(c, 0.0) + 1.0
                    self.global_c_pos[c] = self.global_c_pos.get(c, 0.0) + y
                    self.fold_c_count.setdefault(fold, {})
                    self.fold_c_pos.setdefault(fold, {})
                    self.fold_c_count[fold][c] = self.fold_c_count[fold].get(c, 0.0) + 1.0
                    self.fold_c_pos[fold][c] = self.fold_c_pos[fold].get(c, 0.0) + y

    def _build_similarity(self) -> None:
        self.question_items = sorted(self.global_q_count.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
        self.q_to_idx = {q: i for i, q in enumerate(self.question_items)}
        n = len(self.question_items)
        self.global_q_count_vec = np.zeros(n, dtype=float)
        self.global_q_pos_vec = np.zeros(n, dtype=float)
        self.fold_q_count_vecs: Dict[int, np.ndarray] = {}
        self.fold_q_pos_vecs: Dict[int, np.ndarray] = {}
        for q, idx in self.q_to_idx.items():
            self.global_q_count_vec[idx] = self.global_q_count[q]
            self.global_q_pos_vec[idx] = self.global_q_pos[q]
        for fold in self.folds:
            cvec = np.zeros(n, dtype=float)
            pvec = np.zeros(n, dtype=float)
            for q, v in self.fold_q_count.get(fold, {}).items():
                cvec[self.q_to_idx[q]] = v
            for q, v in self.fold_q_pos.get(fold, {}).items():
                pvec[self.q_to_idx[q]] = v
            self.fold_q_count_vecs[fold] = cvec
            self.fold_q_pos_vecs[fold] = pvec
        self.support_q_count_vecs = {f: self.global_q_count_vec - self.fold_q_count_vecs[f] for f in self.folds}
        self.support_q_pos_vecs = {f: self.global_q_pos_vec - self.fold_q_pos_vecs[f] for f in self.folds}

        if self.question_embedding_path:
            emb_lookup = load_embedding_lookup(self.question_embedding_path)
            dim = None
            rows = []
            for q in self.question_items:
                arr = emb_lookup.get(str(q))
                if arr is None:
                    if dim is None:
                        dim = len(next(iter(emb_lookup.values())))
                    rows.append(np.zeros(dim, dtype=float))
                else:
                    arr = np.asarray(arr, dtype=float)
                    if dim is None:
                        dim = arr.shape[0]
                    rows.append(arr)
            mat = np.vstack(rows)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms <= 0] = 1.0
            mat = mat / norms
            sim = mat @ mat.T
            sim = np.maximum(sim, self.embedding_min_value)
            self.sim_matrix = sim
        else:
            self.sim_matrix = np.eye(n, dtype=float)

    def support_question_stats(self, fold: int, q: str) -> Tuple[float, float]:
        count = self.global_q_count.get(q, 0.0) - self.fold_q_count.get(fold, {}).get(q, 0.0)
        pos = self.global_q_pos.get(q, 0.0) - self.fold_q_pos.get(fold, {}).get(q, 0.0)
        return float(count), float(pos)

    def support_concept_stats(self, fold: int, c: str) -> Tuple[float, float]:
        count = self.global_c_count.get(c, 0.0) - self.fold_c_count.get(fold, {}).get(c, 0.0)
        pos = self.global_c_pos.get(c, 0.0) - self.fold_c_pos.get(fold, {}).get(c, 0.0)
        return float(count), float(pos)

    def support_semantic_stats(self, fold: int, q: str) -> Tuple[float, float]:
        idx = self.q_to_idx.get(str(q))
        if idx is None:
            return 0.0, 0.0
        row = self.sim_matrix[idx]
        total = float(row @ self.support_q_count_vecs[fold])
        pos = float(row @ self.support_q_pos_vecs[fold])
        return total, pos

    def history_semantic_stats(
        self,
        history_questions: Sequence[str],
        history_responses: Sequence[int],
        query_q: str,
        decay_lambda: float,
    ) -> Tuple[float, float]:
        idx = self.q_to_idx.get(str(query_q))
        if idx is None or len(history_questions) == 0:
            return 0.0, 0.0
        exponents = np.arange(len(history_questions) - 1, -1, -1, dtype=float)
        temporal = np.power(decay_lambda, exponents)
        temporal = temporal / temporal.sum()
        sims = np.array(
            [self.sim_matrix[idx, self.q_to_idx[hq]] if hq in self.q_to_idx else 0.0 for hq in history_questions],
            dtype=float,
        )
        weights = temporal * sims
        total = float(weights.sum())
        pos = float(np.dot(weights, np.asarray(history_responses, dtype=float)))
        return total, pos


def _recent_mean(values: Sequence[int], k: int) -> float:
    if len(values) == 0:
        return 0.5
    tail = values[-k:] if len(values) >= k else values
    return float(np.mean(tail))


def _extract_instances_from_sequence_df(
    df: pd.DataFrame,
    support_bank: FeatureSupportBank,
    support_fold: int,
    item_col: str,
    decay_lambda: float,
    include_question_meta: bool = False,
    use_future_context: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray]:
    rows: List[Dict[str, Any]] = []
    labels: List[int] = []

    for row_index, row in df.iterrows():
        questions = _parse_seq(row["questions"]) if "questions" in df.columns else []
        concepts = _parse_seq(row["concepts"]) if "concepts" in df.columns else []
        items = _parse_seq(row[item_col])
        responses = _parse_seq(row["responses"])
        selectmasks = _parse_seq(row["selectmasks"]) if "selectmasks" in row else []
        qidxs = _parse_seq(row["qidxs"]) if "qidxs" in row else []
        orirows = _parse_seq(row["orirow"]) if "orirow" in row else []
        seq_len = min(len(items), len(responses))
        if questions:
            seq_len = min(seq_len, len(questions))
        if concepts:
            seq_len = min(seq_len, len(concepts))
        if selectmasks:
            seq_len = min(seq_len, len(selectmasks))
        if qidxs:
            seq_len = min(seq_len, len(qidxs))
        if orirows:
            seq_len = min(seq_len, len(orirows))

        valid_questions: List[str] = []
        valid_concepts: List[str] = []
        valid_items: List[str] = []
        valid_responses: List[int] = []
        valid_selects: List[int] = []
        valid_qidxs: List[int] = []
        valid_orirows: List[int] = []
        valid_timestamps: List[int] = []
        timestamps = _parse_seq(row["timestamps"]) if "timestamps" in row else []

        for pos in range(seq_len):
            item = str(items[pos])
            resp = responses[pos]
            if item == "-1" or resp == "-1":
                continue
            q = str(questions[pos]) if questions else item
            c = str(concepts[pos]) if concepts else item
            valid_items.append(item)
            valid_questions.append(q)
            valid_concepts.append(c)
            valid_responses.append(_safe_int(resp))
            valid_selects.append(_safe_int(selectmasks[pos], default=1) if selectmasks else 1)
            valid_qidxs.append(_safe_int(qidxs[pos]) if qidxs else -1)
            valid_orirows.append(_safe_int(orirows[pos]) if orirows else int(row_index))
            valid_timestamps.append(_safe_int(timestamps[pos]) if timestamps and pos < len(timestamps) else -1)

        q_hist_count: Dict[str, int] = {}
        q_hist_pos: Dict[str, int] = {}
        c_hist_count: Dict[str, int] = {}
        c_hist_pos: Dict[str, int] = {}
        last_q_pos: Dict[str, int] = {}
        last_c_pos: Dict[str, int] = {}

        future_q_count: Dict[str, int] = {}
        future_q_pos: Dict[str, int] = {}
        future_c_count: Dict[str, int] = {}
        future_c_pos: Dict[str, int] = {}
        next_q_pos: Dict[str, int] = {}
        next_c_pos: Dict[str, int] = {}
        if use_future_context:
            for pos in range(len(valid_items) - 1, -1, -1):
                q = valid_questions[pos]
                c = valid_concepts[pos]
                future_q_count[q] = future_q_count.get(q, 0) + 1
                future_q_pos[q] = future_q_pos.get(q, 0) + valid_responses[pos]
                future_c_count[c] = future_c_count.get(c, 0) + 1
                future_c_pos[c] = future_c_pos.get(c, 0) + valid_responses[pos]
                next_q_pos[q] = pos
                next_c_pos[c] = pos

        suffix_sum = None
        if use_future_context:
            suffix_sum = np.zeros(len(valid_items) + 1, dtype=float)
            for pos in range(len(valid_items) - 1, -1, -1):
                suffix_sum[pos] = suffix_sum[pos + 1] + valid_responses[pos]

        for pos in range(len(valid_items)):
            if use_future_context:
                cur_q = valid_questions[pos]
                cur_c = valid_concepts[pos]
                future_q_count[cur_q] -= 1
                future_q_pos[cur_q] -= valid_responses[pos]
                future_c_count[cur_c] -= 1
                future_c_pos[cur_c] -= valid_responses[pos]
                if future_q_count[cur_q] <= 0:
                    future_q_count.pop(cur_q, None)
                    future_q_pos.pop(cur_q, None)
                    next_q_pos.pop(cur_q, None)
                elif next_q_pos.get(cur_q) == pos:
                    for nxt in range(pos + 1, len(valid_items)):
                        if valid_questions[nxt] == cur_q:
                            next_q_pos[cur_q] = nxt
                            break
                if future_c_count[cur_c] <= 0:
                    future_c_count.pop(cur_c, None)
                    future_c_pos.pop(cur_c, None)
                    next_c_pos.pop(cur_c, None)
                elif next_c_pos.get(cur_c) == pos:
                    for nxt in range(pos + 1, len(valid_items)):
                        if valid_concepts[nxt] == cur_c:
                            next_c_pos[cur_c] = nxt
                            break

            if pos == 0:
                q = valid_questions[pos]
                c = valid_concepts[pos]
                y = valid_responses[pos]
            else:
                prev_q = valid_questions[pos - 1]
                prev_c = valid_concepts[pos - 1]
                prev_y = valid_responses[pos - 1]
                q_hist_count[prev_q] = q_hist_count.get(prev_q, 0) + 1
                q_hist_pos[prev_q] = q_hist_pos.get(prev_q, 0) + prev_y
                c_hist_count[prev_c] = c_hist_count.get(prev_c, 0) + 1
                c_hist_pos[prev_c] = c_hist_pos.get(prev_c, 0) + prev_y
                last_q_pos[prev_q] = pos - 1
                last_c_pos[prev_c] = pos - 1

            if pos == 0 or valid_selects[pos] != 1:
                continue

            q = valid_questions[pos]
            c = valid_concepts[pos]
            y = valid_responses[pos]
            history_responses = valid_responses[:pos]
            hist_len = pos
            hist_correct = float(sum(history_responses))
            hist_acc = _safe_float_div(hist_correct, hist_len)
            same_q_count = float(q_hist_count.get(q, 0))
            same_q_pos = float(q_hist_pos.get(q, 0))
            same_c_count = float(c_hist_count.get(c, 0))
            same_c_pos = float(c_hist_pos.get(c, 0))

            support_q_count, support_q_pos = support_bank.support_question_stats(support_fold, q)
            support_c_count, support_c_pos = support_bank.support_concept_stats(support_fold, c)
            support_sem_count, support_sem_pos = support_bank.support_semantic_stats(support_fold, q)
            hist_sem_count, hist_sem_pos = support_bank.history_semantic_stats(
                valid_questions[:pos], history_responses, q, decay_lambda
            )

            ts = valid_timestamps[pos]
            prev_ts = valid_timestamps[pos - 1] if pos > 0 else -1
            gap_prev = float(max(0, ts - prev_ts) / 60000.0) if ts >= 0 and prev_ts >= 0 else -1.0
            gap_same_q = float(pos - last_q_pos[q]) if q in last_q_pos else -1.0
            gap_same_c = float(pos - last_c_pos[c]) if c in last_c_pos else -1.0

            feat = {
                "hist_len_log": _clip_log1p(hist_len),
                "hist_acc": hist_acc,
                "recent3_acc": _recent_mean(history_responses, 3),
                "recent5_acc": _recent_mean(history_responses, 5),
                "recent10_acc": _recent_mean(history_responses, 10),
                "same_q_count_log": _clip_log1p(same_q_count),
                "same_q_acc": _safe_float_div(same_q_pos, same_q_count),
                "same_c_count_log": _clip_log1p(same_c_count),
                "same_c_acc": _safe_float_div(same_c_pos, same_c_count),
                "support_q_count_log": _clip_log1p(support_q_count),
                "support_q_acc": _safe_float_div(support_q_pos, support_q_count),
                "support_c_count_log": _clip_log1p(support_c_count),
                "support_c_acc": _safe_float_div(support_c_pos, support_c_count),
                "support_sem_count_log": _clip_log1p(support_sem_count),
                "support_sem_acc": _safe_float_div(support_sem_pos, support_sem_count),
                "hist_sem_count_log": _clip_log1p(hist_sem_count),
                "hist_sem_acc": _safe_float_div(hist_sem_pos, hist_sem_count),
                "gap_prev_log": _clip_log1p(gap_prev) if gap_prev >= 0 else -1.0,
                "gap_same_q_log": _clip_log1p(gap_same_q) if gap_same_q >= 0 else -1.0,
                "gap_same_c_log": _clip_log1p(gap_same_c) if gap_same_c >= 0 else -1.0,
            }
            if use_future_context:
                future_len = len(valid_items) - pos - 1
                future_correct = float(suffix_sum[pos + 1]) if suffix_sum is not None else 0.0
                future_acc = _safe_float_div(future_correct, future_len)
                future_rs = valid_responses[pos + 1 :]
                future_q_count_cur = float(future_q_count.get(q, 0))
                future_q_pos_cur = float(future_q_pos.get(q, 0))
                future_c_count_cur = float(future_c_count.get(c, 0))
                future_c_pos_cur = float(future_c_pos.get(c, 0))
                gap_next_q = float(next_q_pos[q] - pos) if q in next_q_pos else -1.0
                gap_next_c = float(next_c_pos[c] - pos) if c in next_c_pos else -1.0
                feat.update(
                    {
                        "future_len_log": _clip_log1p(future_len),
                        "future_acc": future_acc,
                        "future3_acc": _recent_mean(future_rs[:3], 3) if future_len > 0 else 0.5,
                        "future5_acc": _recent_mean(future_rs[:5], 5) if future_len > 0 else 0.5,
                        "future10_acc": _recent_mean(future_rs[:10], 10) if future_len > 0 else 0.5,
                        "future_same_q_count_log": _clip_log1p(future_q_count_cur),
                        "future_same_q_acc": _safe_float_div(future_q_pos_cur, future_q_count_cur),
                        "future_same_c_count_log": _clip_log1p(future_c_count_cur),
                        "future_same_c_acc": _safe_float_div(future_c_pos_cur, future_c_count_cur),
                        "gap_next_q_log": _clip_log1p(gap_next_q) if gap_next_q >= 0 else -1.0,
                        "gap_next_c_log": _clip_log1p(gap_next_c) if gap_next_c >= 0 else -1.0,
                    }
                )
            if include_question_meta:
                feat.update(
                    {
                        "qidx": valid_qidxs[pos],
                        "orirow": valid_orirows[pos],
                        "y_true": y,
                        "query_question": q,
                        "query_concept": c,
                    }
                )
            rows.append(feat)
            labels.append(y)

    return pd.DataFrame(rows), np.asarray(labels, dtype=int)


@dataclass
class NonKTCeilingResult:
    fold_metrics: pd.DataFrame
    question_predictions: pd.DataFrame
    concept_predictions: pd.DataFrame
    summary: Dict[str, Any]


def run_nonkt_ceiling_proxy(
    train_sequence_csv: str,
    test_question_csv: str,
    output_dir: str,
    folds: Sequence[int],
    question_embedding_path: Optional[str] = None,
    decay_lambda: float = 0.9,
    embedding_min_value: float = 0.0,
    max_iter: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    use_future_context: bool = False,
) -> NonKTCeilingResult:
    support_bank = FeatureSupportBank(
        sequence_csv=train_sequence_csv,
        folds=folds,
        question_embedding_path=question_embedding_path,
        embedding_min_value=embedding_min_value,
    )
    train_df = pd.read_csv(train_sequence_csv)
    test_q_df = pd.read_csv(test_question_csv)

    all_fold_metrics: List[Dict[str, Any]] = []
    all_question_rows: List[Dict[str, Any]] = []
    all_concept_rows: List[Dict[str, Any]] = []

    feature_cols = None
    for fold in folds:
        train_sub = train_df[train_df["fold"] != int(fold)].copy()
        X_train_df, y_train = _extract_instances_from_sequence_df(
            train_sub,
            support_bank=support_bank,
            support_fold=int(fold),
            item_col="questions",
            decay_lambda=decay_lambda,
            use_future_context=use_future_context,
        )
        feature_cols = list(X_train_df.columns)
        X_train = X_train_df.to_numpy(dtype=float)

        clf = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_depth=max_depth,
            min_samples_leaf=50,
            random_state=42 + int(fold),
        )
        clf.fit(X_train, y_train)

        X_test_df, y_test = _extract_instances_from_sequence_df(
            test_q_df,
            support_bank=support_bank,
            support_fold=int(fold),
            item_col="questions",
            decay_lambda=decay_lambda,
            include_question_meta=True,
            use_future_context=use_future_context,
        )
        meta_cols = ["qidx", "orirow", "y_true", "query_question", "query_concept"]
        X_eval = X_test_df[[c for c in X_test_df.columns if c not in meta_cols]].to_numpy(dtype=float)
        p = clf.predict_proba(X_eval)[:, 1]
        concept_df = X_test_df[meta_cols].copy()
        concept_df["fold"] = int(fold)
        concept_df["p_correct"] = p
        concept_df["liu_plugin"] = [-(pp * math.log(max(min(pp, 1 - 1e-12), 1e-12)) + (1 - pp) * math.log(max(min(1 - pp, 1 - 1e-12), 1e-12))) for pp in p]
        all_concept_rows.extend(concept_df.to_dict(orient="records"))

        concept_metrics = _classification_metrics(y_test.astype(int), p.astype(float))
        fold_row = {
            "fold": int(fold),
            "concept_auc": concept_metrics["auc"],
            "concept_acc": concept_metrics["acc"],
            "concept_nll": concept_metrics["nll"],
            "num_concept_points": int(len(concept_df)),
            "num_train_points": int(len(X_train_df)),
        }

        qrows = []
        for qidx, group in concept_df.groupby("qidx", sort=True):
            preds = group["p_correct"].tolist()
            y_true_q = int(round(float(group["y_true"].mean())))
            fused = _late_fusion_scores(preds)
            qrow = {
                "fold": int(fold),
                "qidx": int(qidx),
                "orirow": int(group["orirow"].iloc[0]),
                "y_true": y_true_q,
                "num_concepts": int(len(group)),
                "concept_preds": ",".join([str(round(float(x), 6)) for x in preds]),
            }
            qrow.update(fused)
            qrows.append(qrow)
        qdf = pd.DataFrame(qrows)
        all_question_rows.extend(qdf.to_dict(orient="records"))
        for key in ["late_mean", "late_vote", "late_all"]:
            m = _classification_metrics(qdf["y_true"].to_numpy(dtype=int), qdf[key].to_numpy(dtype=float))
            fold_row[f"{key}_auc"] = m["auc"]
            fold_row[f"{key}_acc"] = m["acc"]
            fold_row[f"{key}_nll"] = m["nll"]
        fold_row["num_questions"] = int(len(qdf))
        all_fold_metrics.append(fold_row)

    fold_metrics_df = pd.DataFrame(all_fold_metrics)
    concept_pred_df = pd.DataFrame(all_concept_rows)
    question_pred_df = pd.DataFrame(all_question_rows)
    summary = {
        "train_sequence_csv": train_sequence_csv,
        "test_question_csv": test_question_csv,
        "question_embedding_path": question_embedding_path,
        "folds": [int(f) for f in folds],
        "decay_lambda": float(decay_lambda),
        "embedding_min_value": float(embedding_min_value),
        "max_iter": int(max_iter),
        "max_depth": int(max_depth),
        "learning_rate": float(learning_rate),
        "use_future_context": bool(use_future_context),
        "num_eval_concept_points_total": int(len(concept_pred_df)),
        "num_eval_questions_total": int(len(question_pred_df)),
        "feature_cols": feature_cols or [],
    }
    for col in [c for c in fold_metrics_df.columns if c != "fold"]:
        summary[f"{col}_mean"] = float(fold_metrics_df[col].mean())
        summary[f"{col}_std"] = float(fold_metrics_df[col].std(ddof=0))

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    fold_metrics_df.to_csv(outdir / "fold_metrics.csv", index=False)
    concept_pred_df.to_csv(outdir / "concept_predictions.csv", index=False)
    question_pred_df.to_csv(outdir / "question_predictions.csv", index=False)
    with open(outdir / "summary.json", "w", encoding="utf8") as fout:
        json.dump(summary, fout, ensure_ascii=False, indent=2)

    return NonKTCeilingResult(
        fold_metrics=fold_metrics_df,
        question_predictions=question_pred_df,
        concept_predictions=concept_pred_df,
        summary=summary,
    )

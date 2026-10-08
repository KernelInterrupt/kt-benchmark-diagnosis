from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import math
import numpy as np


KernelFn = Callable[[Any, Any], float]


def binary_entropy(p: float, eps: float = 1e-12) -> float:
    """Binary entropy in nats."""
    p = float(np.clip(p, eps, 1.0 - eps))
    return -p * math.log(p) - (1.0 - p) * math.log(1.0 - p)


def inverse_binary_entropy(h: float, tol: float = 1e-10, max_iter: int = 200) -> float:
    """
    Invert binary entropy on the monotone branch [0, 1/2].

    Returns the local Bayes error floor corresponding to entropy h.
    """
    h = float(h)
    if h <= 0.0:
        return 0.0
    max_h = math.log(2.0)
    if h >= max_h:
        return 0.5

    left, right = 0.0, 0.5
    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        mid_h = binary_entropy(mid)
        if abs(mid_h - h) <= tol:
            return mid
        if mid_h < h:
            left = mid
        else:
            right = mid
    return 0.5 * (left + right)


def exact_match_kernel(item_a: Any, item_b: Any) -> float:
    return 1.0 if item_a == item_b else 0.0


def cosine_kernel_from_dict(
    embedding_dict: Dict[Any, Sequence[float]],
    min_value: float = 0.0,
) -> KernelFn:
    """
    Build a cosine-similarity kernel from a qid -> embedding dictionary.

    Similarities are clipped below by ``min_value`` to keep kernel weights nonnegative.
    """

    def _kernel(item_a: Any, item_b: Any) -> float:
        vec_a = embedding_dict.get(item_a)
        vec_b = embedding_dict.get(item_b)
        if vec_a is None or vec_b is None:
            return 0.0
        a = np.asarray(vec_a, dtype=float)
        b = np.asarray(vec_b, dtype=float)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom <= 0.0:
            return 0.0
        sim = float(np.dot(a, b) / denom)
        return max(min_value, sim)

    return _kernel


@dataclass
class LIUStepEstimate:
    step_index: int
    query_item: Any
    target_response: Optional[int]
    p_correct: float
    p_incorrect: float
    liu_plugin: float
    liu_miller_madow: float
    bayes_error_plugin: float
    bayes_error_miller_madow: float
    total_kernel_weight: float
    effective_sample_size: float
    positive_weight: float
    negative_weight: float
    occupied_classes: int


class LIUEstimator:
    """
    Minimal Local Irreducible Uncertainty (LIU) estimator.

    Core design:
    - compute a local estimate at each time step;
    - then average those local quantities across time or by item.

    The estimator uses:
    - exponential forgetting with decay ``decay_lambda``
    - a nonnegative item kernel ``kernel_fn``
    - additive smoothing ``alpha``
    """

    def __init__(
        self,
        decay_lambda: float = 1.0,
        alpha: float = 1.0,
        kernel_fn: Optional[KernelFn] = None,
    ) -> None:
        if not (0.0 < decay_lambda <= 1.0):
            raise ValueError("decay_lambda must be in (0, 1].")
        if alpha < 0.0:
            raise ValueError("alpha must be nonnegative.")
        self.decay_lambda = float(decay_lambda)
        self.alpha = float(alpha)
        self.kernel_fn = kernel_fn or exact_match_kernel

    def _temporal_weights(self, history_len: int) -> np.ndarray:
        if history_len <= 0:
            return np.array([], dtype=float)
        exponents = np.arange(history_len - 1, -1, -1, dtype=float)
        weights = np.power(self.decay_lambda, exponents)
        denom = float(weights.sum())
        return weights / denom if denom > 0.0 else weights

    def estimate_step(
        self,
        history_items: Sequence[Any],
        history_responses: Sequence[int],
        query_item: Any,
        step_index: int,
    ) -> LIUStepEstimate:
        if len(history_items) != len(history_responses):
            raise ValueError("history_items and history_responses must have the same length.")
        if any(r not in (0, 1) for r in history_responses):
            raise ValueError("history_responses must contain only 0/1 values.")

        history_len = len(history_items)
        temporal_weights = self._temporal_weights(history_len)
        kernel_weights = np.array(
            [max(0.0, float(self.kernel_fn(item, query_item))) for item in history_items],
            dtype=float,
        )
        combined_weights = temporal_weights * kernel_weights
        total_weight = float(combined_weights.sum())

        responses = np.asarray(history_responses, dtype=float)
        positive_weight = float(np.dot(combined_weights, responses)) if history_len > 0 else 0.0
        negative_weight = max(0.0, total_weight - positive_weight)

        p_correct = (positive_weight + self.alpha) / (total_weight + 2.0 * self.alpha)
        p_incorrect = 1.0 - p_correct
        liu_plugin = binary_entropy(p_correct)

        if total_weight > 0.0:
            normalized = combined_weights / total_weight
            n_eff = float(1.0 / np.sum(np.square(normalized)))
        else:
            n_eff = 0.0

        occupied = int(positive_weight > 0.0) + int(negative_weight > 0.0)
        mm_correction = ((occupied - 1) / (2.0 * n_eff)) if occupied > 0 and n_eff > 0.0 else 0.0
        liu_mm = min(math.log(2.0), liu_plugin + mm_correction)

        return LIUStepEstimate(
            step_index=step_index,
            query_item=query_item,
            target_response=None,
            p_correct=float(p_correct),
            p_incorrect=float(p_incorrect),
            liu_plugin=float(liu_plugin),
            liu_miller_madow=float(liu_mm),
            bayes_error_plugin=float(inverse_binary_entropy(liu_plugin)),
            bayes_error_miller_madow=float(inverse_binary_entropy(liu_mm)),
            total_kernel_weight=float(total_weight),
            effective_sample_size=float(n_eff),
            positive_weight=float(positive_weight),
            negative_weight=float(negative_weight),
            occupied_classes=int(occupied),
        )

    def estimate_sequence(
        self,
        items: Sequence[Any],
        responses: Sequence[int],
        min_history: int = 1,
    ) -> List[LIUStepEstimate]:
        """
        Estimate LIU for each observed interaction using all previous interactions as history.

        For a sequence (C_1, R_1), ..., (C_T, R_T), the estimate at output index j
        corresponds to predicting R_j using history up to j-1 and query item C_j.
        """
        if len(items) != len(responses):
            raise ValueError("items and responses must have the same length.")
        if min_history < 0:
            raise ValueError("min_history must be nonnegative.")

        estimates: List[LIUStepEstimate] = []
        for j in range(len(items)):
            if j < min_history:
                continue
            est = self.estimate_step(
                history_items=items[:j],
                history_responses=responses[:j],
                query_item=items[j],
                step_index=j,
            )
            est.target_response = int(responses[j])
            estimates.append(est)
        return estimates


def average_bayes_floor(
    estimates: Sequence[LIUStepEstimate],
    use_miller_madow: bool = True,
) -> float:
    """
    Average Bayes floor over time.

    Important: this averages the per-step local floor, rather than applying h^{-1}
    after averaging entropies.
    """
    if len(estimates) == 0:
        return float("nan")
    values = [
        est.bayes_error_miller_madow if use_miller_madow else est.bayes_error_plugin
        for est in estimates
    ]
    return float(np.mean(values))


def average_liu(
    estimates: Sequence[LIUStepEstimate],
    use_miller_madow: bool = True,
) -> float:
    if len(estimates) == 0:
        return float("nan")
    values = [est.liu_miller_madow if use_miller_madow else est.liu_plugin for est in estimates]
    return float(np.mean(values))


def aggregate_by_item(
    estimates: Sequence[LIUStepEstimate],
    use_miller_madow: bool = True,
) -> Dict[Any, Dict[str, float]]:
    """
    Item-level aggregation for RU and item-level average Bayes floor.
    """
    grouped: Dict[Any, List[LIUStepEstimate]] = {}
    for est in estimates:
        grouped.setdefault(est.query_item, []).append(est)

    summary: Dict[Any, Dict[str, float]] = {}
    for item, group in grouped.items():
        if use_miller_madow:
            liu_values = [g.liu_miller_madow for g in group]
            be_values = [g.bayes_error_miller_madow for g in group]
        else:
            liu_values = [g.liu_plugin for g in group]
            be_values = [g.bayes_error_plugin for g in group]
        summary[item] = {
            "count": float(len(group)),
            "ru": float(np.mean(liu_values)),
            "item_bayes_floor": float(np.mean(be_values)),
            "avg_effective_sample_size": float(np.mean([g.effective_sample_size for g in group])),
            "avg_total_kernel_weight": float(np.mean([g.total_kernel_weight for g in group])),
        }
    return summary


def estimates_to_rows(estimates: Iterable[LIUStepEstimate]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for est in estimates:
        rows.append(
            {
                "step_index": est.step_index,
                "query_item": est.query_item,
                "target_response": est.target_response,
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
            }
        )
    return rows

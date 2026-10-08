from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple


@dataclass
class ContextMixPredictionStats:
    p_correct: float
    deepest_match_depth: int
    deepest_match_total: int
    deepest_match_positive: int
    deepest_match_negative: int


class ContextMixBinaryKTContextEstimator:
    """
    Lightweight binary context-mixing predictor.

    Experts are suffix contexts from depth 0..max_order. Their KT-smoothed
    probabilities are combined with adaptive softmax weights that are updated
    online to favor experts that better explained the observed target.
    """

    def __init__(self, max_order: int = 6, learning_rate: float = 0.5) -> None:
        if max_order < 0:
            raise ValueError("max_order must be non-negative.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        self.max_order = int(max_order)
        self.learning_rate = float(learning_rate)
        self.counts: Dict[Tuple[str, ...], Tuple[int, int]] = {}
        self.weight_logits = [0.0] * (self.max_order + 1)
        self._rollback_logs: list[tuple[list[tuple[Tuple[str, ...], Tuple[int, int] | None]], list[float]]] = []

    def clone(self) -> "ContextMixBinaryKTContextEstimator":
        cloned = ContextMixBinaryKTContextEstimator(
            max_order=self.max_order,
            learning_rate=self.learning_rate,
        )
        cloned.counts = self.counts.copy()
        cloned.weight_logits = list(self.weight_logits)
        return cloned

    def checkpoint(self) -> int:
        return len(self._rollback_logs)

    def rollback(self, marker: int) -> None:
        if marker < 0 or marker > len(self._rollback_logs):
            raise ValueError("rollback marker out of range.")
        while len(self._rollback_logs) > marker:
            count_changes, prev_logits = self._rollback_logs.pop()
            for ctx, previous in reversed(count_changes):
                if previous is None:
                    self.counts.pop(ctx, None)
                else:
                    self.counts[ctx] = previous
            self.weight_logits = prev_logits

    def _suffix(self, context_tokens: Sequence[str]) -> Tuple[str, ...]:
        if self.max_order == 0:
            return tuple()
        if len(context_tokens) <= self.max_order:
            return tuple(context_tokens)
        return tuple(context_tokens[-self.max_order :])

    def _contexts(self, suffix: Tuple[str, ...]) -> list[Tuple[str, ...]]:
        return [suffix[-depth:] if depth > 0 else tuple() for depth in range(0, len(suffix) + 1)]

    def _context_prob(self, ctx: Tuple[str, ...]) -> float:
        count0, count1 = self.counts.get(ctx, (0, 0))
        total = count0 + count1
        return (count1 + 0.5) / (total + 1.0)

    def _softmax_weights(self, count: int) -> list[float]:
        logits = self.weight_logits[:count]
        max_logit = max(logits) if logits else 0.0
        exps = [math.exp(logit - max_logit) for logit in logits]
        denom = sum(exps)
        if denom <= 0.0:
            return [1.0 / max(1, count)] * count
        return [value / denom for value in exps]

    def predict(self, context_tokens: Sequence[str]) -> ContextMixPredictionStats:
        suffix = self._suffix(context_tokens)
        contexts = self._contexts(suffix)
        probs = [self._context_prob(ctx) for ctx in contexts]
        weights = self._softmax_weights(len(contexts))
        p1 = sum(weight * prob for weight, prob in zip(weights, probs))
        p1 = float(min(max(p1, 1e-12), 1.0 - 1e-12))

        deepest_depth = 0
        deepest_count0, deepest_count1 = self.counts.get(tuple(), (0, 0))
        for depth in range(len(suffix), 0, -1):
            ctx = suffix[-depth:]
            count0, count1 = self.counts.get(ctx, (0, 0))
            if count0 + count1 > 0:
                deepest_depth = depth
                deepest_count0, deepest_count1 = count0, count1
                break

        return ContextMixPredictionStats(
            p_correct=p1,
            deepest_match_depth=int(deepest_depth),
            deepest_match_total=int(deepest_count0 + deepest_count1),
            deepest_match_positive=int(deepest_count1),
            deepest_match_negative=int(deepest_count0),
        )

    def update(self, context_tokens: Sequence[str], target: int) -> None:
        if target not in (0, 1):
            raise ValueError("target must be binary.")
        suffix = self._suffix(context_tokens)
        contexts = self._contexts(suffix)
        probs = [self._context_prob(ctx) for ctx in contexts]
        prev_logits = list(self.weight_logits)
        for depth, prob in enumerate(probs):
            correct_prob = prob if target == 1 else (1.0 - prob)
            correct_prob = min(max(correct_prob, 1e-12), 1.0)
            self.weight_logits[depth] += self.learning_rate * math.log(correct_prob)

        changes: list[tuple[Tuple[str, ...], Tuple[int, int] | None]] = []
        for ctx in contexts:
            previous = self.counts.get(ctx)
            changes.append((ctx, previous))
            count0, count1 = previous if previous is not None else (0, 0)
            if target == 1:
                count1 += 1
            else:
                count0 += 1
            self.counts[ctx] = (count0, count1)
        self._rollback_logs.append((changes, prev_logits))


def create_contextmix_estimator(
    max_order: int = 6,
    learning_rate: float = 0.5,
    backend: str = "python",
):
    backend = str(backend).strip().lower()
    if backend == "python":
        return ContextMixBinaryKTContextEstimator(max_order=max_order, learning_rate=learning_rate)
    if backend == "cpp":
        from .contextmix_cpp import CppContextMixBinaryKTContextEstimator

        return CppContextMixBinaryKTContextEstimator(max_order=max_order, learning_rate=learning_rate)
    raise ValueError("backend must be one of: python, cpp")

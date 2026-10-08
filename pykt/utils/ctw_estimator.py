from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Sequence

import math


_LOG_HALF = math.log(0.5)


def _logsumexp_pair(a: float, b: float) -> float:
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


@dataclass
class CTWPredictionStats:
    p_correct: float
    deepest_match_depth: int
    deepest_match_total: int
    deepest_match_positive: int
    deepest_match_negative: int


class _ContextTreeNode:
    def __init__(self) -> None:
        self.count0 = 0
        self.count1 = 0
        self.log_pe = 0.0
        self.log_pw = 0.0
        self.child_log_pw_sum = 0.0
        self.children: Dict[str, "_ContextTreeNode"] = {}

    def _local_predictive_prob(self, target: int) -> float:
        total = self.count0 + self.count1
        if target == 1:
            return (self.count1 + 0.5) / (total + 1.0)
        return (self.count0 + 0.5) / (total + 1.0)

    def _recompute_log_pw(self, depth_remaining: int) -> None:
        if depth_remaining <= 0 or len(self.children) == 0:
            self.log_pw = self.log_pe
            return
        self.log_pw = _logsumexp_pair(_LOG_HALF + self.log_pe, _LOG_HALF + self.child_log_pw_sum)

    def update(self, context_suffix_rev: Sequence[str], target: int, depth_remaining: int) -> None:
        self.log_pe += math.log(self._local_predictive_prob(target))
        if target == 1:
            self.count1 += 1
        else:
            self.count0 += 1

        if depth_remaining > 0 and len(context_suffix_rev) > 0:
            token = context_suffix_rev[0]
            child = self.children.get(token)
            old_child_log_pw = child.log_pw if child is not None else 0.0
            if child is None:
                child = _ContextTreeNode()
                self.children[token] = child
            child.update(context_suffix_rev[1:], target, depth_remaining - 1)
            self.child_log_pw_sum += child.log_pw - old_child_log_pw

        self._recompute_log_pw(depth_remaining)

    def hypothetical_log_pw(
        self,
        context_suffix_rev: Sequence[str],
        target: int,
        depth_remaining: int,
    ) -> float:
        log_pe_after = self.log_pe + math.log(self._local_predictive_prob(target))
        has_child_extension = depth_remaining > 0 and len(context_suffix_rev) > 0
        has_children_after = len(self.children) > 0 or has_child_extension
        if depth_remaining <= 0 or not has_children_after:
            return log_pe_after

        child_log_pw_sum_after = self.child_log_pw_sum
        if has_child_extension:
            token = context_suffix_rev[0]
            child = self.children.get(token)
            old_child_log_pw = child.log_pw if child is not None else 0.0
            if child is None:
                child_log_pw_after = _empty_hypothetical_log_pw(
                    context_suffix_rev[1:],
                    target,
                    depth_remaining - 1,
                )
            else:
                child_log_pw_after = child.hypothetical_log_pw(
                    context_suffix_rev[1:],
                    target,
                    depth_remaining - 1,
                )
            child_log_pw_sum_after += child_log_pw_after - old_child_log_pw

        return _logsumexp_pair(_LOG_HALF + log_pe_after, _LOG_HALF + child_log_pw_sum_after)


def _empty_hypothetical_log_pw(
    context_suffix_rev: Sequence[str],
    target: int,
    depth_remaining: int,
) -> float:
    log_pe_after = math.log(0.5)
    if depth_remaining <= 0 or len(context_suffix_rev) == 0:
        return log_pe_after
    child_after = _empty_hypothetical_log_pw(context_suffix_rev[1:], target, depth_remaining - 1)
    return _logsumexp_pair(_LOG_HALF + log_pe_after, _LOG_HALF + child_after)


class CTWBinaryKTContextEstimator:
    """
    Context Tree Weighting style binary predictor for KT.

    Each prediction conditions on a symbolic suffix context built from
    past interaction tokens plus a dedicated query token for the current item.
    """

    def __init__(self, max_depth: int = 6) -> None:
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1.")
        self.max_depth = int(max_depth)
        self.root = _ContextTreeNode()

    def clone(self) -> "CTWBinaryKTContextEstimator":
        return copy.deepcopy(self)

    def _suffix_rev(self, context_tokens: Sequence[str]) -> Sequence[str]:
        if len(context_tokens) <= self.max_depth:
            return list(reversed(context_tokens))
        return list(reversed(context_tokens[-self.max_depth :]))

    def predict(self, context_tokens: Sequence[str]) -> CTWPredictionStats:
        suffix_rev = self._suffix_rev(context_tokens)
        log_pw_before = self.root.log_pw
        log_pw_after_0 = self.root.hypothetical_log_pw(suffix_rev, target=0, depth_remaining=self.max_depth)
        log_pw_after_1 = self.root.hypothetical_log_pw(suffix_rev, target=1, depth_remaining=self.max_depth)

        raw0 = math.exp(log_pw_after_0 - log_pw_before)
        raw1 = math.exp(log_pw_after_1 - log_pw_before)
        denom = raw0 + raw1
        p1 = raw1 / denom if denom > 0.0 else 0.5

        deepest_node = self.root
        deepest_depth = 0
        node = self.root
        for depth, token in enumerate(suffix_rev, start=1):
            child = node.children.get(token)
            if child is None:
                break
            deepest_node = child
            deepest_depth = depth
            node = child

        pos = int(deepest_node.count1)
        neg = int(deepest_node.count0)
        return CTWPredictionStats(
            p_correct=float(p1),
            deepest_match_depth=int(deepest_depth),
            deepest_match_total=int(pos + neg),
            deepest_match_positive=int(pos),
            deepest_match_negative=int(neg),
        )

    def update(self, context_tokens: Sequence[str], target: int) -> None:
        if target not in (0, 1):
            raise ValueError("target must be binary.")
        suffix_rev = self._suffix_rev(context_tokens)
        self.root.update(suffix_rev, target=target, depth_remaining=self.max_depth)


def create_ctw_estimator(max_depth: int = 6, backend: str = "python"):
    backend = str(backend).strip().lower()
    if backend == "python":
        return CTWBinaryKTContextEstimator(max_depth=max_depth)
    if backend == "cpp":
        from .ctw_cpp import CppCTWBinaryKTContextEstimator

        return CppCTWBinaryKTContextEstimator(max_depth=max_depth)
    raise ValueError("backend must be one of: python, cpp")


def interaction_context_token(item: str, response: int) -> str:
    return f"I:{item}|R:{int(response)}"


def query_context_token(item: str) -> str:
    return f"Q:{item}"

#!/usr/bin/env python3
"""Reproduce the synthetic CTW anchor sanity check (paper Fig. 2)."""
import math
import random
import importlib.util
from collections import deque
from pathlib import Path
import sys

RELEASE_ROOT = Path(__file__).resolve().parent.parent
if str(RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT))

_CTW_MODULE_PATH = RELEASE_ROOT / "pykt" / "utils" / "ctw_estimator.py"
_SPEC = importlib.util.spec_from_file_location("release_ctw_estimator", _CTW_MODULE_PATH)
_CTW_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _CTW_MODULE
_SPEC.loader.exec_module(_CTW_MODULE)
CTWBinaryKTContextEstimator = _CTW_MODULE.CTWBinaryKTContextEstimator
interaction_context_token = _CTW_MODULE.interaction_context_token
query_context_token = _CTW_MODULE.query_context_token

DEPTH = 6
N_SEQUENCES = 64
SEQUENCE_LENGTH = 400
SEEDS = (0, 1, 42)


def iid_sequence(rng):
    return [rng.randint(0, 1) for _ in range(SEQUENCE_LENGTH)]


def persistent_sequence(rng):
    state = rng.randint(0, 1)
    seq = []
    for _ in range(SEQUENCE_LENGTH):
        seq.append(state)
        if rng.random() >= 0.95:
            state = 1 - state
    return seq


def evaluate(sequences):
    entropies, correct = [], []
    for seq in sequences:
        estimator = CTWBinaryKTContextEstimator(max_depth=DEPTH)
        history = deque(maxlen=DEPTH - 1)
        for step, response in enumerate(seq):
            context = list(history) + [query_context_token("c")]
            if step >= 1:  # one-step warm-up, as in the reported protocol
                p = estimator.predict(context).p_correct
                entropies.append(-(p * math.log2(p) + (1 - p) * math.log2(1 - p))
                                 if 0 < p < 1 else 0.0)
                correct.append(float((p >= 0.5) == (response == 1)))
            estimator.update(context, int(response))
            history.append(interaction_context_token("c", int(response)))
    return sum(entropies) / len(entropies), sum(correct) / len(correct)


def main():
    collected = {"iid": [], "persistent": []}
    for seed in SEEDS:
        rng = random.Random(seed)
        iid = [iid_sequence(rng) for _ in range(N_SEQUENCES)]
        rng = random.Random(seed + 1000)
        persistent = [persistent_sequence(rng) for _ in range(N_SEQUENCES)]
        i_h, i_a = evaluate(iid)
        p_h, p_a = evaluate(persistent)
        collected["iid"].append((i_h, i_a))
        collected["persistent"].append((p_h, p_a))
        print(f"seed={seed:<3d} iid: H={i_h:.3f} bits acc={i_a:.3f} | "
              f"persistent: H={p_h:.3f} bits acc={p_a:.3f}")
    print("mean across seeds: " + " | ".join(
        f"{name}: H={sum(x[0] for x in rows)/len(rows):.3f} bits, "
        f"acc={sum(x[1] for x in rows)/len(rows):.3f}"
        for name, rows in collected.items()))


if __name__ == "__main__":
    main()

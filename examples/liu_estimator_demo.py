import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pykt.utils.ctw_estimator import (
    CTWBinaryKTContextEstimator,
    interaction_context_token,
    query_context_token,
)
from pykt.utils.liu_estimator import (
    LIUEstimator,
    aggregate_by_item,
    average_bayes_floor,
    average_liu,
    binary_entropy,
    cosine_kernel_from_dict,
    estimates_to_rows,
)


def main():
    # Toy sequential interaction log.
    items = ["q1", "q2", "q1", "q3", "q2", "q1", "q3"]
    responses = [1, 0, 1, 0, 1, 0, 1]

    # Frozen item semantics; replace with qwen3-vl-embedding or other cached features in practice.
    frozen_embeddings = {
        "q1": [1.0, 0.1, 0.0],
        "q2": [0.8, 0.2, 0.1],
        "q3": [0.0, 0.9, 0.8],
    }

    estimator = LIUEstimator(
        decay_lambda=0.9,
        alpha=1.0,
        kernel_fn=cosine_kernel_from_dict(frozen_embeddings, min_value=0.0),
    )

    estimates = estimator.estimate_sequence(items, responses, min_history=1)

    print("== per-step LIU estimates ==")
    for row in estimates_to_rows(estimates):
        print(row)

    print("\n== aggregate metrics ==")
    print("Average LIU:", average_liu(estimates))
    print("Average Bayes floor:", average_bayes_floor(estimates))
    print("Per-item summary:", aggregate_by_item(estimates))

    print("\n== CTW-style symbolic predictions ==")
    ctw = CTWBinaryKTContextEstimator(max_depth=4)
    history_tokens = []
    for idx, (item, response) in enumerate(zip(items, responses)):
        context = history_tokens + [query_context_token(item)]
        if idx > 0:
            pred = ctw.predict(context)
            print(
                {
                    "step_index": idx,
                    "query_item": item,
                    "p_correct": round(pred.p_correct, 6),
                    "liu_plugin": round(binary_entropy(pred.p_correct), 6),
                    "deepest_match_depth": pred.deepest_match_depth,
                    "deepest_match_total": pred.deepest_match_total,
                }
            )
        ctw.update(context, response)
        history_tokens.append(interaction_context_token(item, response))


if __name__ == "__main__":
    main()

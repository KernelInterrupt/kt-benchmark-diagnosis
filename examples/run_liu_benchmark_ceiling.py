import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pykt.utils.liu_benchmark_runner import run_folded_question_ceiling


def _parse_folds(text: str):
    if not text.strip():
        return [0, 1, 2, 3, 4]
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Run fold-aware, fusion-aware LIU benchmark ceiling.")
    parser.add_argument("--support_csv", type=str, required=True, help="Typically train_valid_sequences.csv")
    parser.add_argument("--eval_question_csv", type=str, required=True, help="Typically test_question_sequences.csv")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--folds", type=str, default="0,1,2,3,4")
    parser.add_argument("--item_col", type=str, default="questions")
    parser.add_argument("--decay_lambda", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--history_weight", type=float, default=1.0)
    parser.add_argument("--embedding_path", type=str, default="")
    parser.add_argument("--embedding_min_value", type=float, default=0.0)
    parser.add_argument("--estimator", type=str, default="kernel", help="Conditional distribution estimator: kernel or ctw")
    parser.add_argument("--ctw_max_depth", type=int, default=6, help="Maximum symbolic suffix depth for CTW estimator")
    parser.add_argument("--ctw_backend", type=str, default="python", help="CTW backend: python or cpp")
    args = parser.parse_args()

    result = run_folded_question_ceiling(
        support_csv=args.support_csv,
        eval_question_csv=args.eval_question_csv,
        output_dir=args.output_dir,
        folds=_parse_folds(args.folds),
        item_col=args.item_col,
        decay_lambda=args.decay_lambda,
        alpha=args.alpha,
        history_weight=args.history_weight,
        embedding_path=args.embedding_path or None,
        embedding_min_value=args.embedding_min_value,
        estimator=args.estimator,
        ctw_max_depth=args.ctw_max_depth,
        ctw_backend=args.ctw_backend,
    )

    print("== summary ==")
    for k, v in result.summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

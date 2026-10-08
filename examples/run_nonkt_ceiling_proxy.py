import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pykt.utils.nonkt_ceiling_proxy import run_nonkt_ceiling_proxy


def _parse_folds(text: str):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Run non-KT cross-fitted ceiling proxy.")
    parser.add_argument("--train_sequence_csv", type=str, required=True)
    parser.add_argument("--test_question_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--folds", type=str, default="0,1,2,3,4")
    parser.add_argument("--question_embedding_path", type=str, default="")
    parser.add_argument("--decay_lambda", type=float, default=0.9)
    parser.add_argument("--embedding_min_value", type=float, default=0.0)
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--use_future_context", type=int, default=0, help="If 1, use bidirectional/offline oracle features from suffix context")
    args = parser.parse_args()

    result = run_nonkt_ceiling_proxy(
        train_sequence_csv=args.train_sequence_csv,
        test_question_csv=args.test_question_csv,
        output_dir=args.output_dir,
        folds=_parse_folds(args.folds),
        question_embedding_path=args.question_embedding_path or None,
        decay_lambda=args.decay_lambda,
        embedding_min_value=args.embedding_min_value,
        max_iter=args.max_iter,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        use_future_context=bool(args.use_future_context),
    )
    print("== summary ==")
    for k, v in result.summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

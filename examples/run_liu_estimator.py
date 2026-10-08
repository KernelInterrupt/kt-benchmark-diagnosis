import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pykt.utils.liu_sequence_runner import run_liu_on_processed_csv, save_sequence_run_result


def _parse_folds(text: str):
    if not text.strip():
        return None
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Run LIU estimator on pyKT processed sequence CSV.")
    parser.add_argument("--sequence_csv", type=str, required=True, help="Path to processed sequence CSV, e.g. train_valid_sequences_quelevel.csv")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save per-step, per-item, and summary outputs")
    parser.add_argument("--decay_lambda", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--min_history", type=int, default=1)
    parser.add_argument("--folds", type=str, default="", help="Optional comma-separated folds, e.g. 0,1,2")
    parser.add_argument("--score_only_selected", type=int, default=1, help="If 1, only output steps with selectmask==1")
    parser.add_argument("--item_col", type=str, default="", help="Sequence column to use as item ids: questions or concepts. Defaults to auto-detect.")
    parser.add_argument("--embedding_path", type=str, default="", help="Optional frozen qid embedding file (.pt/.pth/.npy/.json)")
    parser.add_argument("--embedding_min_value", type=float, default=0.0, help="Clip cosine kernel below this value")
    parser.add_argument("--estimator", type=str, default="kernel", help="Conditional distribution estimator: kernel or ctw")
    parser.add_argument("--ctw_max_depth", type=int, default=6, help="Maximum symbolic suffix depth for CTW estimator")
    parser.add_argument("--ctw_backend", type=str, default="python", help="CTW backend: python or cpp")
    args = parser.parse_args()

    result = run_liu_on_processed_csv(
        sequence_csv=args.sequence_csv,
        decay_lambda=args.decay_lambda,
        alpha=args.alpha,
        min_history=args.min_history,
        folds=_parse_folds(args.folds),
        score_only_selected=bool(args.score_only_selected),
        item_col=args.item_col or None,
        embedding_path=args.embedding_path or None,
        embedding_min_value=args.embedding_min_value,
        estimator=args.estimator,
        ctw_max_depth=args.ctw_max_depth,
        ctw_backend=args.ctw_backend,
    )
    saved = save_sequence_run_result(result, args.output_dir)

    print("== LIU summary ==")
    for key, value in result.summary.items():
        print(f"{key}: {value}")

    print("\n== saved files ==")
    for key, value in saved.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

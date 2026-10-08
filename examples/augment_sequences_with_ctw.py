import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pykt.utils.ctw_feature_builder import augment_sequence_csv_with_ctw


def build_parser():
    parser = argparse.ArgumentParser(description="Augment a processed sequence CSV with CTW per-step features.")
    parser.add_argument("--sequence_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--item_col", type=str, default="questions")
    parser.add_argument("--support_csv", type=str, default="", help="Optional support CSV. If omitted, build out-of-fold CTW from sequence_csv itself.")
    parser.add_argument("--ctw_max_depth", type=int, default=6)
    parser.add_argument("--ctw_backend", type=str, default="python")
    return parser


def main():
    args = build_parser().parse_args()

    out = augment_sequence_csv_with_ctw(
        sequence_csv=args.sequence_csv,
        output_csv=args.output_csv,
        item_col=args.item_col,
        support_csv=args.support_csv or None,
        max_depth=args.ctw_max_depth,
        backend=args.ctw_backend,
    )
    print(out)


if __name__ == "__main__":
    main()

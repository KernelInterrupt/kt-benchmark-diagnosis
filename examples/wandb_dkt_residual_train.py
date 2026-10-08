import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from wandb_train import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2015")
    parser.add_argument("--model_name", type=str, default="dkt_residual")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--emb_size", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--early_stop_patience", type=int, default=4)
    parser.add_argument("--ctw_feat_dim", type=int, default=64)
    parser.add_argument("--ctw_delta_scale", type=float, default=1.0)
    parser.add_argument("--train_valid_file_override", type=str, default="")
    parser.add_argument("--test_file_override", type=str, default="")
    parser.add_argument("--test_window_file_override", type=str, default="")
    parser.add_argument("--test_question_file_override", type=str, default="")
    parser.add_argument("--test_question_window_file_override", type=str, default="")
    parser.add_argument("--selection_metric", type=str, default="validnll")
    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--add_uuid", type=int, default=1)
    args = parser.parse_args()
    main(vars(args))

import os
import pandas as pd
import numpy as np

SCALAR_COLUMNS = {"fold", "uid", "dataset"}
SEQUENCE_COLUMNS = {
    "questions",
    "concepts",
    "responses",
    "timestamps",
    "usetimes",
    "selectmasks",
    "is_repeat",
    "qidxs",
    "rest",
    "orirow",
    "cidxs",
}


def _split_seq(value):
    if pd.isna(value):
        return []
    tokens = str(value).split(",")
    if len(tokens) > 0 and tokens[-1] == "":
        tokens = tokens[:-1]
    return tokens


def _join_seq(tokens):
    return ",".join([str(t) for t in tokens])


def _pad_seq(tokens, target_len, pad_token="-1"):
    if len(tokens) >= target_len:
        return tokens[:target_len]
    return tokens + [pad_token] * (target_len - len(tokens))

def apply_sparsity(train_df, ratio=1.0, seed=42):
    """Samples a portion of the training data."""
    if ratio >= 1.0:
        return train_df
    print(f"Applying Sparsity: Training on {ratio*100}% of data (seed {seed})")
    return train_df.sample(frac=ratio, random_state=seed)

def get_cold_start_stats(train_df, test_df):
    """
    Identifies which questions in the test set were never seen in the training set.
    Returns a set of seen question IDs.
    """
    seen_qids = set()
    # Assuming standard pykt format where 'questions' is a comma-separated string
    for qs in train_df['questions'].dropna():
        seen_qids.update(str(qs).split(','))
    
    print(f"Cold Start Analysis: Found {len(seen_qids)} unique questions in training set.")
    return seen_qids


def build_question_holdout_set(train_df, holdout_ratio=0.25, seed=42):
    """
    Builds a deterministic held-out question set from the train split only.
    """
    all_qids = set()
    for qs in train_df["questions"].dropna():
        all_qids.update([q for q in _split_seq(qs) if q != "-1"])

    all_qids = sorted(all_qids)
    if len(all_qids) == 0:
        print("Question Holdout: no valid questions found in training split.")
        return set()

    holdout_num = max(1, int(round(len(all_qids) * holdout_ratio)))
    holdout_num = min(holdout_num, len(all_qids))
    rng = np.random.default_rng(seed)
    heldout_qids = set(rng.choice(np.array(all_qids, dtype=object), size=holdout_num, replace=False).tolist())
    print(
        f"Question Holdout: sampled {len(heldout_qids)} / {len(all_qids)} training questions "
        f"(ratio={holdout_ratio}, seed={seed})"
    )
    return heldout_qids


def _filter_row_train_seen_only(row, heldout_qids, min_seq_len=3):
    questions = _split_seq(row.get("questions", ""))
    if len(questions) == 0:
        return None

    original_len = len(questions)
    keep_mask = [(q not in heldout_qids) and (q != "-1") for q in questions]
    if sum(keep_mask) < min_seq_len:
        return None

    new_row = row.copy()
    for col in row.index:
        if col in SCALAR_COLUMNS or col not in SEQUENCE_COLUMNS:
            continue
        values = _split_seq(row[col])
        if len(values) == 0:
            continue
        if len(values) != len(questions):
            continue
        filtered_values = [v for v, keep in zip(values, keep_mask) if keep]
        filtered_values = _pad_seq(filtered_values, original_len, pad_token="-1")
        new_row[col] = _join_seq(filtered_values)
    return new_row


def _filter_row_eval_targets_only(row, heldout_qids):
    questions = _split_seq(row.get("questions", ""))
    selectmasks = _split_seq(row.get("selectmasks", ""))
    if len(questions) == 0 or len(selectmasks) == 0 or len(questions) != len(selectmasks):
        return None

    new_masks = []
    for idx, (q, mask) in enumerate(zip(questions, selectmasks)):
        if idx == 0:
            new_masks.append("0")
        elif mask == "1" and q in heldout_qids:
            new_masks.append("1")
        else:
            new_masks.append("0")

    if sum(int(m) for m in new_masks[1:]) == 0:
        return None

    new_row = row.copy()
    new_row["selectmasks"] = _join_seq(new_masks)
    return new_row


def apply_question_holdout_split(df, heldout_qids, train_folds, eval_folds, min_seq_len=3):
    """
    Applies a held-out-question protocol:
    - train folds: remove held-out question interactions from sequences
    - eval folds: keep full sequences but only score targets whose question is held out
    """
    processed_rows = []
    train_folds = set(train_folds)
    eval_folds = set(eval_folds)

    for _, row in df.iterrows():
        fold = row["fold"]
        if fold in train_folds:
            new_row = _filter_row_train_seen_only(row, heldout_qids, min_seq_len=min_seq_len)
        elif fold in eval_folds:
            new_row = _filter_row_eval_targets_only(row, heldout_qids)
        else:
            new_row = row

        if new_row is not None:
            processed_rows.append(new_row)

    if len(processed_rows) == 0:
        return pd.DataFrame(columns=df.columns)
    return pd.DataFrame(processed_rows, columns=df.columns)


def apply_question_holdout_eval(df, heldout_qids):
    """
    Keeps test/window-test sequences intact, but only evaluates positions whose target question
    belongs to the held-out set.
    """
    processed_rows = []
    for _, row in df.iterrows():
        new_row = _filter_row_eval_targets_only(row, heldout_qids)
        if new_row is not None:
            processed_rows.append(new_row)
    if len(processed_rows) == 0:
        return pd.DataFrame(columns=df.columns)
    return pd.DataFrame(processed_rows, columns=df.columns)

def filter_test_scenarios(test_df, seen_qids, scenario="standard"):
    """
    Filters or marks the test dataframe based on the requested scenario.
    """
    if scenario == "standard":
        return test_df
    
    def is_cold(qs):
        q_list = str(qs).split(',')
        # A sequence is 'cold' if it contains at least one unseen question
        # Or we could define it as 'all questions are unseen'. 
        # Usually, we want to evaluate per-interaction.
        return any(q not in seen_qids for q in q_list)

    if scenario == "question_cold_start":
        print("Scenario: Question Cold Start (filtering for sequences with unseen questions)")
        # This is a simple filter. For fine-grained evaluation, 
        # we'd usually add a mask to the batch in the DataLoader.
        mask = test_df['questions'].apply(is_cold)
        return test_df[mask]
    
    return test_df

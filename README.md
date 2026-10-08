# CTW anchor and knowledge-tracing analysis tools

This package contains the symbolic Context Tree Weighting (CTW) anchor, the LIU uncertainty estimator, neural residual-probe training entry points, and analysis/export code for knowledge tracing. Use Python >= 3.10.

## Anchor information set

The shipped anchor conditions on the last `D-1` pairs of (item identifier, binary response), plus the currently queried item identifier. It uses no skill tags and no Q-matrix. This is implemented by [`pykt/utils/ctw_estimator.py`](pykt/utils/ctw_estimator.py) and [`pykt/utils/ctw_feature_builder.py`](pykt/utils/ctw_feature_builder.py).

## Analysis tools

The `scripts/` and `tools/` directories contain analysis and export workflows for aligned predictions, entropy bands, item-level diagnostics, symbolic benchmark artifacts, and residual probes. They require external benchmark data and/or prediction exports; those inputs are not included. Use each command's `--help` options to supply input and output paths. Where supported, paths can be relocated with `PYKT_REPO_ROOT`.

The analysis workflows include:

- `scripts/rebuttal_unified_analysis.py` for common-cohort metrics, entropy bands, calibration, and fold-level tests.
- `scripts/rebuttal_postprocess.py` for prediction postprocessing and figures.
- `tools/export_full_paper_tables.py` for derived table exports, including item-level RU/IG summaries.
- `tools/build_strict_symbolic_fold_jobs.py` and `tools/export_strict_symbolic_question_predictions.py` for strict symbolic benchmark artifacts.
- `tools/run_simplekt_residual_local_bayes_worker.py` and related queue builders for residual-probe training/evaluation.

## Running the anchor

Install the package and dependencies in a Python >= 3.10 environment. The CTW anchor can be exercised with `python examples/liu_estimator_demo.py` or `python examples/run_liu_benchmark_ceiling.py`. Neural probe training uses the existing W&B entry point, for example:

```sh
python examples/wandb_simplekt_residual_train.py --dataset_name nips_task34 --model_name simplekt_residual --emb_type qid --fold 0
```

Supply the dataset, fold artifacts, and training parameters used for your experiment. W&B examples expect a local `configs/wandb.json`; copy `configs/wandb.json.example` there and fill it locally. `configs/best_model.json.example` shows the dataset/model/checkpoint-list keys expected by the merge utility. Never commit actual credentials.

`python examples/fig2_sanity_check.py` runs the synthetic CTW sanity-check program. `python examples/env_check.py` reports locally available dependencies and command usability.

## Included files

The repository includes code, configuration templates, and documentation. It does not include benchmark datasets, run exports, checkpoints, per-sample predictions, or derived result tables. Results can be exported by the provided tools when their required inputs are available.

## Provenance

This archive is a fork of the upstream project `pykt-toolkit` (`github.com/pykt-team/pykt-toolkit`), with `setup.py` version `0.0.38`.

The inherited defects described in the source README are not repaired in this archive. In particular, `examples/extract_raw_result.py:522-526`, `examples/extract_quelevel_raw_result.py:222-223`, and `pykt/models/evaluate_model.py:1031-1037` contain hard-coded `/root/autodl-nas/...` paths. The single exception is `examples/generate_ab_wandb.py`, which carries a one-line change so the launcher no longer prints the user's W&B API key.

## Limitations and dependencies

`pykt/utils/_ctw_core.so` is a Linux x86-64 build. Rebuild on other platforms with `examples/build_ctw_cpp.py`; its rebuild check compares modification times only. The anchor's maximum context depth `D` counts a token budget of `(D-1)` past (item, response) tokens plus the current query token, so effective response-history depth is `D-1`.

CUDA/cuDF alignment and support scripts require NVIDIA RAPIDS; the residual worker requires `torch` and scikit-learn. Run `python examples/env_check.py` for a local dependency report. `requirements.txt` lists the package dependencies.

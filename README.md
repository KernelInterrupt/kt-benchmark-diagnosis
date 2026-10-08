# CTW anchor and knowledge-tracing analysis tools

This source distribution provides the symbolic Context Tree Weighting (CTW) anchor, the LIU uncertainty estimator, neural residual-probe training entry points, and analysis/export code for knowledge tracing. Use Python >= 3.10.

## Layout

- `pykt/` contains the reusable library, models, preprocessing code, and estimators.
- `examples/` contains runnable demonstrations, training/evaluation launchers, and data-preparation helpers.
- `scripts/` and `tools/` contain analysis and export workflows.
- `configs/` contains configuration templates.

Run example and analysis commands from the source tree (a checkout or an extracted source distribution). They read benchmark data and prediction exports from paths supplied by the caller; use each command's `--help` option for its inputs. Where supported, paths can be relocated with `PYKT_REPO_ROOT`.

## Anchor information set

The CTW anchor conditions on the last `D-1` pairs of (item identifier, binary response), plus the currently queried item identifier. It uses no skill tags and no Q-matrix. This is implemented by [`pykt/utils/ctw_estimator.py`](pykt/utils/ctw_estimator.py) and [`pykt/utils/ctw_feature_builder.py`](pykt/utils/ctw_feature_builder.py).

## Installation and use

Install the package and its dependencies in a Python >= 3.10 environment:

```sh
python -m pip install .
```

The CTW/LIU demo can be run with `python examples/liu_estimator_demo.py`. The synthetic CTW sanity check is `python examples/fig2_sanity_check.py`. Neural probe training uses the W&B entry point, for example:

```sh
python examples/wandb_simplekt_residual_train.py --dataset_name nips_task34 --model_name simplekt_residual --emb_type qid --fold 0
```

Supply the dataset, fold artifacts, and training parameters for the workflow being run. W&B examples expect a local `configs/wandb.json`; copy `configs/wandb.json.example` there and fill it locally. `configs/best_model.json.example` shows the dataset/model/checkpoint-list keys expected by the merge utility. Never commit actual credentials. `python examples/env_check.py` reports locally available dependencies and command usability.

## Analysis entry points

- `scripts/rebuttal_unified_analysis.py` provides cohort metrics, entropy bands, calibration, and fold-level tests.
- `scripts/rebuttal_postprocess.py` provides prediction postprocessing and figures.
- `tools/export_full_paper_tables.py` exports derived tables, including item-level RU/IG summaries.
- `tools/build_strict_symbolic_fold_jobs.py` and `tools/export_strict_symbolic_question_predictions.py` build strict symbolic benchmark artifacts.
- `tools/run_simplekt_residual_local_bayes_worker.py` and its queue builders support residual-probe training/evaluation.

## Provenance and dependencies

This archive is based on the upstream [`pykt-toolkit`](https://github.com/pykt-team/pykt-toolkit) project, with package version `0.0.38`. It is distributed under the MIT license.

The source-tree `pykt/utils/_ctw_core.so` is a Linux x86-64 build. The wheel carries the portable C++ source instead; build the optional C++ backend locally with `python examples/build_ctw_cpp.py` when needed. The anchor's maximum context depth `D` counts `(D-1)` past (item, response) tokens plus the current query token, so effective response-history depth is `D-1`.

CUDA/cuDF alignment and support scripts require NVIDIA RAPIDS; the residual worker requires `torch` and scikit-learn. `requirements.txt` records the development environment pins, while package installation uses the dependencies declared in `setup.py`.

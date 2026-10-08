# An Information-Theoretic Evaluation Framework for Benchmark and Model Diagnosis in Knowledge Tracing

**Houru Jiang, Zixi Wang, Tengteng Cheng, Xueyi Li, Mingliang Hou, Jiaqi Zheng, Renqiang Luo, Teng Guo, Zitao Liu**
**NeurIPS 2026** · [Official NeurIPS poster](https://neurips.cc/virtual/2026/loc/paris/poster/149407)

## Overview

Knowledge-tracing (KT) systems are usually compared with aggregate AUC or accuracy. Those global scores hide where errors originate and do not show whether a benchmark is close to saturation. The project treats local predictability as the diagnostic object instead.

The framework uses Context Tree Weighting (CTW) as an operational causal anchor to estimate the Local Irreducible Uncertainty (LIU) of each student interaction. Model predictions are projected onto this shared uncertainty coordinate, and gains are evaluated across entropy bands rather than only with a single global score. In this release, the CTW context is built from past item/response tokens and the current query-item token; no skill tags or Q-matrix are required by the anchor implementation.

***REMOVED***

***REMOVED***

## What is in this repository

- [`pykt/`](pykt/) — reusable KT models, data loaders and preprocessing, plus CTW/LIU estimators in [`pykt/utils/`](pykt/utils/).
- [`examples/`](examples/) — CTW/LIU demos, synthetic sanity checks, sequence augmentation, preprocessing helpers, and W&B training/evaluation entry points.
- [`scripts/`](scripts/) — post-processing and uncertainty-band analyses that consume caller-provided artifacts.
- [`tools/`](tools/) — alignment, export, queue-building, and other analysis utilities for externally produced predictions/checkpoints.
- [`configs/`](configs/) — dataset/model configuration files and safe W&B/best-model templates.

The core anchor is implemented in [`pykt/utils/ctw_estimator.py`](pykt/utils/ctw_estimator.py); [`pykt/utils/ctw_feature_builder.py`](pykt/utils/ctw_feature_builder.py) adds per-step CTW features to a processed sequence CSV.

## Install

Use Python 3.10 or newer:

```sh
python -m pip install .
python examples/env_check.py
```

For editable development installs, use `python -m pip install -e .`. The runtime dependencies are declared in [`setup.py`](setup.py); [`requirements.txt`](requirements.txt) records the development pins.

## Quick use

Run the lightweight demonstrations from the repository root:

```sh
python examples/liu_estimator_demo.py
python examples/fig2_sanity_check.py
```

Augment a processed sequence CSV (with `questions` and `responses` columns) with per-step CTW features. When no support file is supplied, the input must also contain a `fold` column so the helper can build out-of-fold anchors:

```sh
python examples/augment_sequences_with_ctw.py \
  --sequence_csv path/to/sequence.csv \
  --output_csv path/to/sequence_with_ctw.csv \
  --item_col questions \
  --ctw_max_depth 6 \
  --ctw_backend python
```

The optional C++ backend can be built with `python examples/build_ctw_cpp.py`; the checked-in shared object is platform-specific, so the Python backend is the portable fallback. W&B training scripts (for example [`examples/wandb_simplekt_residual_train.py`](examples/wandb_simplekt_residual_train.py)) require your own datasets, run configuration, and credentials. Keep local credentials out of version control; [`configs/wandb.json.example`](configs/wandb.json.example) is only a template.

The analysis scripts accept caller-provided manifests and prediction exports. To inspect the expected inputs without running an analysis, use:

```sh
python scripts/rebuttal_postprocess.py --check-inputs --repo . --datasets nips_task34 algebra2005
```

## Scope and limitations

This repository is a code release. It does not include a `results/` directory, benchmark datasets, trained checkpoints, or per-sample prediction exports. Analysis and export commands therefore read paths supplied by the caller; they are not a turnkey reproduction of every table in the NeurIPS paper. Dataset access, model training, checkpoint selection, and prediction generation must be provided separately. The synthetic sanity check is a software smoke test, not a claim about the paper’s reported numbers.

The package is distributed under the [MIT License](LICENSE) and follows the upstream [`pykt-toolkit`](https://github.com/pykt-team/pykt-toolkit) layout.

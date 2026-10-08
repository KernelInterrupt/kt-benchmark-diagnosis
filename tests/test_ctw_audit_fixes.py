import importlib
import inspect

import pandas as pd
import pytest
import torch

from pykt.models.akt_residual import AKTResidual
from pykt.models.dkt_residual import DKTResidual
from pykt.models.simplekt_residual import SimpleKTResidual
from pykt.utils.ctw_feature_builder import augment_sequence_csv_with_ctw


def _sequence_batch():
    values = {
        "qseqs": [[0, 1]],
        "cseqs": [[0, 1]],
        "rseqs": [[0, 1]],
        "shft_qseqs": [[0, 1]],
        "shft_cseqs": [[0, 1]],
        "shft_rseqs": [[0, 1]],
    }
    return {key: torch.tensor(value) for key, value in values.items()}


def _model_cases():
    return [
        pytest.param(
            SimpleKTResidual(
                n_question=4,
                n_pid=0,
                d_model=8,
                n_blocks=1,
                dropout=0.0,
                d_ff=16,
                seq_len=8,
                num_attn_heads=2,
            ),
            id="simplekt_residual",
        ),
        pytest.param(
            AKTResidual(
                n_question=4,
                n_pid=0,
                d_model=8,
                n_blocks=1,
                dropout=0.0,
                d_ff=16,
                num_attn_heads=2,
            ),
            id="akt_residual",
        ),
        pytest.param(
            DKTResidual(num_c=4, emb_size=8, dropout=0.0, ctw_feat_dim=4),
            id="dkt_residual",
        ),
    ]


def test_ctw_cli_and_builder_default_to_python(tmp_path, monkeypatch):
    cli = importlib.import_module("examples.augment_sequences_with_ctw")
    parser = cli.build_parser()
    assert parser.parse_args(["--sequence_csv", "in.csv", "--output_csv", "out.csv"]).ctw_backend == "python"
    assert inspect.signature(augment_sequence_csv_with_ctw).parameters["backend"].default == "python"

    sequence_csv = pd.DataFrame(
        {
            "questions": ["1,2,3"],
            "responses": ["1,0,1"],
            "fold": [0],
        }
    )
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    sequence_csv.to_csv(input_path, index=False)
    # Keep the test independent of any checked-in platform-specific shared object.
    # The Python backend must complete even if C++ loading is made impossible.
    import pykt.utils.ctw_cpp as ctw_cpp

    monkeypatch.setattr(
        ctw_cpp,
        "_load_ctw_lib",
        lambda: (_ for _ in ()).throw(AssertionError("C++ CTW backend was loaded")),
    )
    augment_sequence_csv_with_ctw(str(input_path), str(output_path))
    augmented = pd.read_csv(output_path)
    assert {"ctw_pseqs", "ctw_logitseqs", "ctw_depthseqs", "ctw_totalseqs", "ctw_posseqs", "ctw_negseqs"} <= set(
        augmented.columns
    )


@pytest.mark.parametrize("model", _model_cases())
def test_residual_models_require_ctw_base_probability(model):
    dcur = _sequence_batch()
    if isinstance(model, DKTResidual):
        dcur = {key: dcur[key] for key in ("cseqs", "rseqs", "shft_cseqs")}
    with pytest.raises(ValueError, match="CTW.*ctw_pseqs"):
        model(dcur)


@pytest.mark.parametrize("model", _model_cases())
def test_residual_models_forward_with_ctw_base_probability(model):
    dcur = _sequence_batch()
    if isinstance(model, DKTResidual):
        dcur = {key: dcur[key] for key in ("cseqs", "rseqs", "shft_cseqs")}
        dcur["shft_ctw_pseqs"] = torch.tensor([[0.25, 0.75]])
        predictions = model(dcur)
        assert predictions.shape == (1, 2)
    else:
        dcur["ctw_pseqs"] = torch.tensor([[0.25, 0.75]])
        dcur["shft_ctw_pseqs"] = torch.tensor([[0.35, 0.65]])
        predictions = model(dcur)
        assert predictions.shape == (1, 3)

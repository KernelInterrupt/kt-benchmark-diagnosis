import pytest

import pykt.utils.symbolic_benchmark_builder as symbolic_builder


def test_symbolic_builder_supports_ctw_and_contextmix(monkeypatch):
    monkeypatch.setattr(symbolic_builder, "create_ctw_estimator", lambda **kwargs: ("ctw", kwargs))
    monkeypatch.setattr(
        symbolic_builder,
        "create_contextmix_estimator",
        lambda **kwargs: ("contextmix", kwargs),
    )

    assert symbolic_builder._method_name(" CTW ") == "ctw"
    assert symbolic_builder._method_name("ContextMix") == "contextmix"
    assert symbolic_builder._method_suffix("ctw", 4) == "ctw_depth4"
    assert symbolic_builder._method_suffix("contextmix", 4) == "contextmix_order4"
    assert symbolic_builder._create_estimator("ctw", 4, "python")[0] == "ctw"
    assert symbolic_builder._create_estimator("contextmix", 4, "python")[0] == "contextmix"


def test_symbolic_builder_rejects_unsupported_method():
    with pytest.raises(ValueError, match="method must be one of: ctw, contextmix"):
        symbolic_builder._method_name("unsupported")

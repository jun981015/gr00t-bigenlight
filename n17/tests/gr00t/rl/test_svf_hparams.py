from gr00t.rl.svf_hparams import resolve_lambda_multiplier
import pytest


@pytest.mark.parametrize(
    "k,g,c", [(1.0, 1.0, 1.0), (0.4, 0.25, 0.64), (0.4, 0.5, 0.32), (0.2, 0.25, 0.16)]
)
def test_k_g_mapping(k, g, c):
    assert resolve_lambda_multiplier(k, g) == pytest.approx(c)


def test_defaults_and_legacy():
    assert resolve_lambda_multiplier(1) == 1
    assert resolve_lambda_multiplier(0.4, legacy_c=2) == 2
    assert resolve_lambda_multiplier(0, legacy_c=1) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(kappa=0, g=0.2),
        dict(kappa=1, g=0),
        dict(kappa=1, g=float("nan")),
        dict(kappa=1, g=0.2, legacy_c=1),
        dict(kappa=1, g=0.2, soft_lambda=0.1),
        dict(kappa=-1),
        dict(kappa=1, legacy_c=0),
    ],
)
def test_invalid(kwargs):
    with pytest.raises(ValueError):
        resolve_lambda_multiplier(**kwargs)

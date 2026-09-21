"""User-facing (kappa, g) controls; checkpoints retain canonical (kappa, c)."""

import math


def resolve_lambda_multiplier(kappa, g=None, legacy_c=None, soft_lambda=None):
    if not math.isfinite(kappa) or kappa < 0:
        raise ValueError("kappa must be finite and nonnegative")
    if g is not None:
        if legacy_c is not None or soft_lambda is not None:
            raise ValueError("--g cannot be combined with --lambda-multiplier or --soft-lambda")
        if not math.isfinite(g) or g <= 0 or kappa <= 0:
            raise ValueError("(kappa,g) parameterization requires both values finite and positive")
        c = kappa**2 / g
    else:
        c = 1.0 if legacy_c is None else legacy_c
    if not math.isfinite(c) or c <= 0:
        raise ValueError("Resolved lambda multiplier must be finite and positive")
    return c

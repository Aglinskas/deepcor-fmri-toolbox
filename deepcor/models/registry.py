"""Model registry for DeepCor models."""

from .cvae import CVAE
from .cvae_v1 import CVAE_V1

# Model registry for easy access to different model versions
#
# Notes
# -----
# - ``'v1'`` maps to the original CVAE implementation (no confound decoder).
# - ``'v2'`` maps to the current confound-aware CVAE in ``cvae.py``.
# - ``'cvae'`` and ``'latest'`` are stable aliases that always point to
#   the most recent recommended model version.
MODEL_REGISTRY = {
    "v1": CVAE_V1,
    "v2": CVAE,
    "cvae": CVAE,
    "latest": CVAE,
}


def get_model(version: str = "cvae", **kwargs):
    """
    Get a model from the registry.

    Args:
        version: Model version identifier
        **kwargs: Arguments to pass to model constructor

    Returns:
        Instantiated model

    Raises:
        ValueError: If model version not found in registry
    """
    if version not in MODEL_REGISTRY:
        raise ValueError(
            f"Model version '{version}' not found. "
            f"Available versions: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[version](**kwargs)


def list_models():
    """List all available model versions."""
    return list(MODEL_REGISTRY.keys())

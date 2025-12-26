"""Model registry for DeepCor models."""

from .cvae import CVAE

# Model registry for easy access to different model versions
MODEL_REGISTRY = {
    'v1': CVAE,
    'v2': CVAE,  # Placeholder for future versions
    'cvae': CVAE,
}


def get_model(version='cvae', **kwargs):
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

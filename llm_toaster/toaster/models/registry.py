"""Model registry and factory."""

from __future__ import annotations

from .transformer import TransformerModel

MODEL_REGISTRY = {"decoder_transformer": TransformerModel}


def build_model(config):
    architecture = config.model.architecture
    if architecture not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model architecture {architecture!r}")
    return MODEL_REGISTRY[architecture](config.model, config.attention)

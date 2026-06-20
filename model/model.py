"""Deprecated compatibility layer for the original top-level model package.

The canonical model now lives in ``llm_toaster.toaster.models.transformer`` and is
built through ``llm_toaster.toaster.models.registry.build_model``. This shim only
remains so external code using the historical ``from model import TransformerModel``
positional constructor keeps working; new code should use the registry.
"""

import warnings

from llm_toaster.toaster.config.schema import AttentionConfig, ModelConfig
from llm_toaster.toaster.models.attention import MultiHeadAttention
from llm_toaster.toaster.models.transformer import TransformerBlock
from llm_toaster.toaster.models.transformer import TransformerModel as _TransformerModel

warnings.warn(
    "model.TransformerModel is deprecated; build models via llm_toaster.toaster.models.registry.build_model instead.",
    DeprecationWarning,
    stacklevel=2,
)


class TransformerModel(_TransformerModel):
    """Compatibility shim preserving the historical constructor signature."""

    def __init__(
        self,
        n_head: int,
        vocab_size: int,
        n_embd: int,
        seq_len: int,
        device: str = "cpu",
        dropout_rate: float = 0.0,
        n_blocks: int = 4,
        decoder: bool = True,
        **_kwargs,
    ):
        model_config = ModelConfig(
            n_head=n_head,
            vocab_size=vocab_size,
            n_embd=n_embd,
            seq_len=seq_len,
            dropout_rate=dropout_rate,
            n_blocks=n_blocks,
        )
        super().__init__(model_config, AttentionConfig())


FlashAttention = MultiHeadAttention
SelfAttention = MultiHeadAttention

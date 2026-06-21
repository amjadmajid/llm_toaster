"""Parameter accounting and a matched-parameter solver for architecture sweeps.

``estimate_params`` mirrors the real model exactly (verified against ``build_model`` in tests),
so the solver can search candidate sizes without instantiating models. ``solve_for_target_params``
adjusts one dimension (width or depth) so different architectures can be compared at *equal*
total parameters rather than equal configs.
"""

from __future__ import annotations

import copy


def estimate_params(config) -> int:
    """Exact total parameter count for ``config`` without building the model."""
    model = config.model
    vocab = model.vocab_size or 50304
    embd, heads, blocks, seq_len = model.n_embd, model.n_head, model.n_blocks, model.seq_len
    kv_heads = model.num_key_value_heads or heads
    head_dim = embd // heads
    kv_width = kv_heads * head_dim
    hidden = model.ffn_mult * embd

    def linear(in_features: int, out_features: int, bias: bool = True) -> int:
        return in_features * out_features + (out_features if bias else 0)

    norm = 2 * embd if model.norm == "layernorm" else embd  # LayerNorm has weight+bias; RMSNorm weight only

    attn = linear(embd, embd) + 2 * linear(embd, kv_width) + linear(embd, embd)  # q, k, v, o
    if model.ffn == "gelu":
        ffn = linear(embd, hidden) + linear(hidden, embd)
    else:  # geglu / swiglu are gated: gate + up + down
        ffn = 2 * linear(embd, hidden) + linear(hidden, embd)
    block = attn + ffn + 2 * norm  # two norms per block

    token_emb = vocab * embd
    pos_emb = seq_len * embd if model.position == "learned" else 0
    final_norm = norm
    lm_head = 0 if model.tie_embeddings else linear(embd, vocab, bias=False)
    return token_emb + pos_emb + blocks * block + final_norm + lm_head


def solve_for_target_params(config, target_params: int, vary: str = "n_embd"):
    """Return a copy of ``config`` with ``vary`` (``n_embd`` or ``n_blocks``) adjusted so the
    total parameter count is as close as possible to ``target_params``.

    ``n_embd`` is stepped by ``n_head`` (×2 when ``position='rope'`` so head_dim stays even);
    ``n_blocks`` by 1. Parameter count is monotonic in both, so a forward scan finds the closest.
    """
    if vary not in {"n_embd", "n_blocks"}:
        raise ValueError("vary must be 'n_embd' or 'n_blocks'")
    result = copy.deepcopy(config)
    step = 1
    if vary == "n_embd":
        step = result.model.n_head * (2 if result.model.position == "rope" else 1)

    best_value, best_params = None, None
    value = step
    max_value = step * 100_000  # safety bound
    while value <= max_value:
        setattr(result.model, vary, value)
        params = estimate_params(result)
        if best_value is None or abs(params - target_params) < abs(best_params - target_params):
            best_value, best_params = value, params
        if params >= target_params:  # monotonic: closest is the just-below or this just-above
            break
        value += step

    setattr(result.model, vary, best_value)
    return result

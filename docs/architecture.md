# Architecture

LLM Toaster trains a **dense, decoder-only Transformer** (GPT/LLaMA-style): pre-norm residual
blocks, causal self-attention, a feed-forward block, tied input/output embeddings by default, and
GPT-2 style weight init (so training starts near `ln(vocab)`). Every architectural choice is a
config field (see the [README options table](../README.md#model-architecture--options)), so a study
can vary **one axis at a time**.

Diagrams below use the **default config** for concrete numbers
(`n_embd=E=1024`, `n_head=8`, `head_dim=128`, `n_blocks=16`, `vocab=50304`, `seq_len=1024`). `B`/`T`
are batch/sequence at runtime. For an exact, auto-generated card for *any* config (with a parameter
table), run:

```bash
python scripts/describe_arch.py --config config/default_config.yaml
```

## Model dataflow

```mermaid
flowchart TD
    X["input_ids (B, T)"] --> TE["token_embeddings<br/>Embedding(50304, 1024) → (B, T, E)"]
    PE["position_embeddings<br/>Embedding(1024, 1024)<br/>(learned only)"] --> ADD(("+"))
    TE --> ADD
    ADD --> BLK["TransformerBlock × 16"]
    BLK --> FN["final norm (E=1024)"]
    FN --> LM["lm_head<br/>Linear(1024, 50304)<br/>(tied to token_embeddings)"]
    LM --> OUT["logits (B, T, 50304)"]
```

With `position: rope` or `none`, the learned `position_embeddings` table is dropped (RoPE injects
position inside attention; `none` uses no positional signal).

## Decoder block (pre-norm, ×16)

```mermaid
flowchart TD
    IN["x (B, T, E)"] --> N1["norm1"]
    N1 --> ATT["MultiHeadAttention"]
    ATT --> R1(("+"))
    IN --> R1
    R1 --> N2["norm2"]
    N2 --> FF["feed-forward"]
    FF --> R2(("+"))
    R1 --> R2
    R2 --> OUT["(B, T, E)"]
```

Residual output projections (attention `o_proj`, FFN down-projection) are flagged
`_is_residual_projection` and initialised with std `0.02 / sqrt(2·n_blocks)` to keep the residual
stream stable with depth.

## Attention: MHA → GQA → MQA (`model.num_key_value_heads`)

All queries use `n_head` heads; **key/value heads are shared** to shrink the KV-cache — the dominant
inference-memory cost on-device. Query heads map onto KV heads in groups.

```mermaid
flowchart LR
    subgraph MHA["MHA — kv_heads = 8"]
        q1[q0..q7] --> kv1[kv0..kv7]
    end
    subgraph GQA["GQA — kv_heads = 2"]
        q2[q0..q7] --> kv2[kv0, kv1]
    end
    subgraph MQA["MQA — kv_heads = 1"]
        q3[q0..q7] --> kv3[kv0]
    end
```

KV-cache per token (fp16) = `2 · n_blocks · kv_heads · head_dim · 2 bytes`. For the default dims:

| variant | `kv_heads` | KV-cache/token | @ seq_len=1024 |
| --- | ---: | ---: | ---: |
| MHA | 8 | 64 KB | 64 MB |
| GQA | 2 | 16 KB | 16 MB |
| MQA | 1 | 8 KB | 8 MB |

**RoPE** (`position: rope`) rotates query/key pairs by position-dependent angles (rotate-half
convention) before attention; it requires an even `head_dim` and adds no parameters.

## Feed-forward variants (`model.ffn`, width `model.ffn_mult · E`)

```mermaid
flowchart LR
    subgraph GELU["gelu (plain MLP)"]
        a["x (E)"] --> a1["Linear E→4E"] --> a2["GELU"] --> a3["Linear 4E→E"]
    end
    subgraph GATED["geglu / swiglu (gated)"]
        b["x (E)"] --> bg["gate: Linear E→4E"]
        b --> bu["up: Linear E→4E"]
        bg --> bact["GELU (geglu) / SiLU (swiglu)"]
        bact --> bm(("×"))
        bu --> bm
        bm --> bd["down: Linear 4E→E"]
    end
```

Gated FFNs (GEGLU/SwiGLU) carry an extra input projection, so at equal `ffn_mult` they have **more
parameters** than the plain GELU MLP. Architecture comparisons therefore equalise *total*
parameters with the matched-parameter solver (`toaster/models/sizing.py`), not equal `ffn_mult`.

## Normalization (`model.norm`)

- **LayerNorm**: per-token mean+variance normalize, `weight` + `bias` (2·E params).
- **RMSNorm**: root-mean-square normalize, `weight` only (E params) — cheaper, no mean subtraction.

## Reference: default config parameter breakdown

254.1M params total (≈ 969 MB fp32 / 485 MB fp16):

| component | params | % |
| --- | ---: | ---: |
| token_embeddings (tied to lm_head) | 51.5M | 20.3% |
| position_embeddings (learned) | 1.0M | 0.4% |
| attention ×16 | 67.2M | 26.4% |
| feed_forward ×16 | 134.3M | 52.9% |
| norms ×16 + final | ~0.07M | 0.0% |
| **total** | **254.1M** | **100%** |

Compute ≈ 1.72 GFLOP/token (training fwd+bwd). For small models the embedding table is a large
share of parameters and memory — a real lever for on-device deployment (vocab size, tied embeddings).

## Not implemented (rejected at config load)

MoE FFN, sliding-window attention, `flash_attn_2`/`xformers` backends, SentencePiece tokenizer, and
non-`none` `distributed.backend` — `ConfigHandler.validate()` raises a clear `NotImplementedError`
rather than silently pretending to support them.

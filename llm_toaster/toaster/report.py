"""Auto-generated architecture "card": Markdown with Mermaid diagrams + a parameter table.

Numbers are derived from the live model/config, so a card is always exact for that config (never
stale). ``scripts/describe_arch.py`` writes one for any config; ``docs/architecture.md`` holds the
hand-authored design overview and links to this generator.
"""

from __future__ import annotations

from .training.metrics import architecture_summary, format_architecture_summary, human_bytes, human_count


def parameter_table(model) -> tuple[list[tuple[str, int]], int]:
    """Per-component parameter counts (tied lm_head reported as 0, shared with token embeddings)."""
    blocks = model.TransformerBlocks
    n_blocks = len(blocks)
    block = blocks[0]
    per_attn = sum(p.numel() for p in block.attention.parameters())
    per_ffn = sum(p.numel() for p in block.feed_forward.parameters())
    per_norm = sum(p.numel() for p in block.norm1.parameters()) + sum(p.numel() for p in block.norm2.parameters())

    tied = model.lm_head.weight is model.token_embeddings.weight
    rows = [("token_embeddings", model.token_embeddings.weight.numel())]
    if model.position_embeddings is not None:
        rows.append(("position_embeddings", model.position_embeddings.weight.numel()))
    rows += [
        (f"attention (×{n_blocks})", per_attn * n_blocks),
        (f"feed_forward (×{n_blocks})", per_ffn * n_blocks),
        (f"norms (×{n_blocks})", per_norm * n_blocks),
        ("final_norm", sum(p.numel() for p in model.norm.parameters())),
        ("lm_head", 0 if tied else model.lm_head.weight.numel()),
    ]
    total = sum(p.numel() for p in model.parameters())
    return rows, total


def _dataflow_mermaid(summary: dict, config) -> str:
    embd, vocab = config.model.n_embd, (config.model.vocab_size or 50304)
    lines = [
        "```mermaid",
        "flowchart TD",
        '    X["input_ids (B, T)"] --> TE["token_embeddings: Embedding(' + f"{vocab}, {embd}" + ')<br/>(B, T, E)"]',
    ]
    if summary["positions"] == "learned":
        lines.append('    PE["position_embeddings: Embedding(T, E)"] --> ADD(("+"))')
        lines.append("    TE --> ADD")
        lines.append(f'    ADD --> BLK["TransformerBlock × {config.model.n_blocks}"]')
    else:
        note = "RoPE inside attention" if summary["positions"] == "rope" else "no positional signal"
        lines.append(f'    TE --> BLK["TransformerBlock × {config.model.n_blocks}<br/>({note})"]')
    lines += [
        f'    BLK --> FN["final {summary["norm"]} (E={embd})"]',
        f'    FN --> LM["lm_head: Linear(E, {vocab})"]',
        '    LM --> OUT["logits (B, T, V)"]',
        "```",
    ]
    return "\n".join(lines)


def _block_mermaid(summary: dict, config) -> str:
    embd = config.model.n_embd
    hidden = config.model.ffn_mult * embd
    attn_label = (
        f"MultiHeadAttention<br/>{summary['attention_kind']} "
        f"q_heads={summary['n_head']}, kv_heads={summary['num_key_value_heads']}, head_dim={summary['head_dim']}"
    )
    if summary["positions"] == "rope":
        attn_label += "<br/>+RoPE on q,k"
    ffn_label = f"{summary['ffn'].upper()} FFN<br/>hidden={config.model.ffn_mult}·E={hidden}"
    return "\n".join(
        [
            "```mermaid",
            "flowchart TD",
            '    IN["x (B, T, E)"] --> N1[' + f"{summary['norm']}" + "]",
            f'    N1 --> ATT["{attn_label}"]',
            '    ATT --> R1(("+"))',
            "    IN --> R1",
            f"    R1 --> N2[{summary['norm']}]",
            f'    N2 --> FF["{ffn_label}"]',
            '    FF --> R2(("+"))',
            "    R1 --> R2",
            '    R2 --> OUT["(B, T, E)"]',
            "```",
        ]
    )


def architecture_report(model, config) -> str:
    """Full Markdown architecture card for ``model`` built from ``config``."""
    summary = architecture_summary(model, config)
    rows, total = parameter_table(model)

    out = [f"# Architecture card: {summary['architecture']}", ""]
    out += [f"- {line}" for line in format_architecture_summary(summary)]
    out += ["", "## Forward dataflow (layers & interfaces)", "", _dataflow_mermaid(summary, config)]
    out += ["", "## Decoder block (×{})".format(config.model.n_blocks), "", _block_mermaid(summary, config)]
    out += ["", "## Parameters by component", "", "| Component | Params | % |", "| --- | ---: | ---: |"]
    for name, count in rows:
        pct = (count / total * 100) if total else 0.0
        out.append(f"| {name} | {count:,} ({human_count(count)}) | {pct:.1f}% |")
    out.append(f"| **total** | **{total:,} ({human_count(total)})** | **100%** |")
    out += ["", f"Weights ≈ {human_bytes(total * 4)} fp32 / {human_bytes(total * 2)} fp16."]
    return "\n".join(out) + "\n"

# Data pipeline

LLM Toaster's canonical pretraining data format is **immutable, pretokenized token shards
described by a versioned JSON manifest**. Downloading, Hugging Face streaming, background
prefetching, and local caching are *data-acquisition policies* on top of that one format — not
separate trainers.

```text
Document source            (Hugging Face stream | local file)
  -> deterministic tokenization & packing   (TokenPacker: EOT delimiter, contiguous packing)
  -> immutable token shards + manifest       (atomic publish: .partial -> fsync -> checksum -> rename -> append)
  -> sampling / batching / resume state       (PretrainBatchSource: exhaustion policy, manifest refresh)
  -> TrainingEngine
```

The trainer only ever talks to a small `PretrainBatchSource` interface
(`llm_toaster/toaster/data/protocol.py`), so prepared and prefetch modes use **exactly the same
trainer-facing manifest/shard contract**, and a future object-store/Streaming adapter can be added
without touching `TrainingEngine`.

---

## Modes

| | **prepared** (default) | **prefetch** (recommended for Colab/spot) | **direct** (experimental) |
| --- | --- | --- | --- |
| When shards are written | fully, before training | by a background producer, ahead of the trainer | never (tokenized in-process) |
| Trainer reads | `ManifestShardSource` | `ManifestShardSource` (wait mode) | `HFDirectTokenSource` |
| Reproducibility | strongest (immutable, checksummed) | strong (same shard contract) | weaker (see resume guarantees) |
| Disk for train shards | full dataset | bounded queue (`prefetch_shards`) | ~none |
| Best for | research, sweeps, HPC, offline | Colab, spot, limited local disk | quick exploration only |

Prepared and prefetch are byte-for-byte the same to the trainer — only *who writes the shards* and
the exhaustion policy differ (`stop`/`repeat` vs. `wait`). Direct mode bypasses shards entirely and
is gated to a single process with `num_workers=0` and no buffered shuffle; **its validation still
uses a fixed, materialized manifest.**

### Recommended choices

- **Prepared** for reproducible experiments and architecture sweeps.
- **Prefetch** for Colab and limited local disk — no need to pre-download the whole dataset.
- **Direct** only for exploratory runs where you accept weaker resume guarantees.

---

## The manifest

`manifest.json` (versioned, `format_version: 1`) separates an **immutable identity** from a
**mutable append generation**:

- `dataset_fingerprint` — `sha256` over the source identity (incl. resolved revision), the
  tokenizer fingerprint, and the transform semantics (`add_eot`/`packing`/`dtype`). Immutable.
- `generation` — strictly increases on every committed update (each sealed shard, each
  `complete` flag). Mutable. A growing prefetch manifest stays **resume-compatible** with an
  earlier checkpoint as long as the committed prefix through the checkpoint's current shard is
  unchanged.

Each shard entry records `id`, `index`, `split`, relative `path`, `tokens`, `dtype`, `bytes`,
`sha256`, `created_at`. Shard IDs are immutable; existing entries are never mutated; updates are
append-only except for top-level status/counter fields. Shards are published atomically (write a
`.partial`, `fsync`, checksum, `os.replace`, append the entry, publish the manifest), so a
consumer never observes a partial shard and a crash leaves no committed entry.

Inspect and verify any manifest:

```bash
python scripts/data.py inspect  --manifest dataspace/fineweb/manifest.json
python scripts/data.py validate --manifest dataspace/fineweb/manifest.json   # checks every checksum
```

---

## Token-budget planning

```text
tokens_per_step = batch_size * seq_len * gradient_accumulation_steps   # n_batches here
```

To train for a target number of **unique** tokens you must materialize at least that many:

```text
required_tokens   = max_tokens                      (or max_iter * tokens_per_step)
shards            = ceil(required_tokens / shard_tokens)
storage (bytes)   = shards * shard_tokens * itemsize  (uint16 -> 2 bytes/token)
optimizer steps   = required_tokens / tokens_per_step
```

`python scripts/data.py prepare --config <cfg> --dry-run` prints exactly this plan.

**Worked example — why four 10M-token shards can't feed a unique-data 100k-step run.**
With `batch_size=4`, `seq_len=1024`, `n_batches=8`, `tokens_per_step = 4·1024·8 = 32,768`.
A 100,000-step run consumes `100,000 · 32,768 ≈ 3.28B` unique tokens. Four 10M shards hold only
`40M` tokens — about `1,220` steps of unique data. The remaining ~98,800 steps would silently
**repeat** the same 40M tokens ~80×. That is why exhaustion is explicit: with the default
`stop`, the run ends at ~1,220 steps and tells you the data ran out; with `repeat`, every new pass
is logged and `unique_tokens_seen` is tracked separately from `tokens_seen`. To actually do 100k
unique-data steps you need ~328 shards of 10M tokens (~6.5 GB at uint16).

---

## Exhaustion

```yaml
data:
  sampling:
    exhaustion: stop   # stop | repeat | wait
```

- **stop** (default) — raise a data-exhaustion signal, end at a clean optimizer-step boundary,
  save a final checkpoint, log that unique data was exhausted. Never silently restarts.
- **repeat** — explicitly restart at the beginning, increment `pass_index`, mark batches repeated,
  log each new pass, and track `unique_tokens_seen` vs total `tokens_seen`. Deterministic per seed.
- **wait** — (prefetch) refresh the manifest and wait for a newly sealed shard; stop/fail when the
  producer reports completion/failure; respects `materialization.wait_timeout_s`. Requires
  `materialization.mode: prefetch` (or `materialization.external_producer: true`).

If exhaustion happens midway through gradient accumulation, the incomplete step is discarded
cleanly: gradients are zeroed, the optimizer and `global_step` do not advance, and the event is
logged.

---

## Validation

Validation is **fixed and reproducible** — it never grows while training runs.

```yaml
data:
  validation:
    manifest_path: null     # use a separate validation manifest, or...
    tokens: 10_000_000      # ...size it by tokens, or...
    shards: null            # ...by shard count (set exactly one when generating)
    reset_each_eval: true   # reset the cursor before each eval so evals are comparable
    sequential: false       # opt in to walking sequentially instead
```

The producer materializes the validation split **first** and marks it `complete` (frozen) before
training begins. By default the validation cursor resets before each evaluation, so successive
evals see the same examples; set `sequential: true` to walk it instead. Evaluation uses the same
mixed-precision autocast as training and never disturbs the training cursor.

There is no `split_ratio` anymore: a tiny shard count can no longer produce a misleading validation
proportion (the old "2% configured, 25% actual" bug).

---

## Resume guarantees

On resume the engine verifies, before training continues, that the current data is compatible with
the checkpoint and rejects a mismatch with a clear error (it never silently applies modulo to an
out-of-range shard index):

- dataset identity (`dataset_id`),
- resolved source revision,
- tokenizer fingerprint,
- transform fingerprint (`add_eot`/`packing`/`dtype`),
- current shard id **and** checksum,
- the committed manifest prefix through the current shard (appended shards beyond it are fine).

**Prepared / prefetch** resume the *exact* next batch (current shard + token offset). **Direct**
mode restores the pending token buffer, records consumed, and (where the installed `datasets`
supports the state API) the HF iterable state — but bitwise-exact resume is *not* claimed; direct
mode prints a prominent experimental warning.

---

## Storage

- **Persistent shard store** (`materialization.store_dir`, or the directory of `manifest_path`):
  holds `manifest.json` + `shards/`. Put it on durable storage (Drive/EBS/`$WORK`).
- **Optional local read cache** (`materialization.cache_dir`): read-through cache that does not
  change manifest identity (reserved hook).
- **`retain_consumed: true`** (default) keeps consumed shards. Exact resume requires the current
  shard to still exist — if you delete consumed shards, you lose exact resume for checkpoints that
  reference them. Do not delete data that retained checkpoints still need.
- **Checkpoint size** is dominated by model+optimizer state; the data state is tiny (cursor +
  fingerprints), except direct mode also stores its pending token buffer.

---

## Migration from the legacy layout

Old runs produced a directory of loose shards with no manifest, e.g.:

```text
dataspace/fineweb/
  shard_000000_train.npy
  shard_000001_train.npy
  shard_000000_val.npy
```

Register them into a manifest **without retokenizing** (files are referenced in place; nothing is
rewritten or moved):

```bash
python scripts/data.py migrate-legacy \
  --data-dir dataspace/fineweb \
  --manifest dataspace/fineweb/manifest.json
```

Then point your config at it:

```yaml
data:
  manifest_path: dataspace/fineweb/manifest.json
  materialization: { mode: prepared, store_dir: dataspace/fineweb }
```

Until you migrate, an **old-style config** (`training.data_dir` / `data.{dataset_name,remote_name,
split_ratio,shard_size,tokenized_data}`) still trains — the engine reads the directory through the
deprecated `LegacyShardDirSource` and prints one deprecation warning. New-style `prepared` configs
that find unmanifested shards stop with the migration command above rather than guessing.

The legacy `python dataspace/src/download_tokenize_hf.py` command still works as a thin wrapper
around `prepare` (append-safe). Its `--stream` flag means *stream the source while materializing
shards* — never trainer-time streaming (that's `materialization.mode: prefetch`).

---

## Dependencies

The core training path is lightweight. Hugging Face `datasets` is an **optional** dependency,
imported lazily only when a `huggingface` source is streamed:

```bash
pip install -e ".[data]"     # datasets>=2.19 (2.19 is the floor for direct-mode iterable state)
```

A `local` source (`source.type: local`, `dataset_name: path/to/corpus.{txt,jsonl}`) needs no extra
dependency and is fully offline — ideal for tests, air-gapped nodes, and small custom corpora.

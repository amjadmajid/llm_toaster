# Changelog

Reliability hardening for small-LM architecture-comparison experiments. Worked in small,
checked stages (see `docs`/plan). Each entry records what changed, why, and how it was verified.

## Stage 1 — Package / import hygiene

**What changed**
- Removed the stale `LLM_Toaster.egg-info/` directory — a leftover from the project-name drift
  `LLM-Toaster` → `llm-toaster`. It was untracked and gitignored (`.gitignore: *.egg-info`); a fresh
  `pip install -e .` regenerates only `llm_toaster.egg-info`. Its `top_level.txt` even listed `tests`,
  which could mislead packaging tools.
- `pyproject.toml`: added a defensive `exclude` (`tests*`, `runs*`, `checkpoints*`, `logs*`, `docs*`,
  `assets*`) to `[tool.setuptools.packages.find]` and a comment explaining why the generic-named legacy
  packages (`config`, `dataspace`, `model`, `tokenizer_lib`, `utils`) intentionally stay discoverable.

**Why**
- Shipping generic top-level names in a built wheel pollutes/shadows site-packages; the duplicate
  egg-info confused tooling. This stage cleans the hygiene **without moving packages**, because the
  engine hard-imports `from dataspace import DataLoaderLite` (`llm_toaster/toaster/training/engine.py`)
  and editable installs use strict PEP 660 (the finder exposes only the discovered packages) — so a
  rename/move is the *riskiest* change and is deferred to a dedicated stage.

**Verified**
- `python -m compileall .` — clean.
- `pip install -e .` — succeeds; only `llm_toaster.egg-info` present afterward; editable finder still
  maps all of `config, dataspace, llm_toaster, model, tokenizer_lib, utils`.
- All import styles work from repo root **and** from an unrelated CWD (editable mode):
  `import llm_toaster`, `from config import ConfigHandler`, `from dataspace import DataLoaderLite`,
  `from model import TransformerModel`, `from tokenizer_lib import gpt2_encode`,
  `from utils import count_parameters`.
- `python trainer.py --help`, `python inference.py --help` — both run.
- `pytest -q` — no regressions.

**Confirmed-issue findings (full analysis)**
Inspected the 10 reported issues against the current tree. The repo has two stacks — a canonical
modular engine (`llm_toaster/toaster/`) and thin legacy top-level shims. Status:
- **Real & actionable:** #1 package hygiene (this stage), #9 atomic checkpoint write + interrupt safety,
  #6 resume (persist elapsed seconds; tokenizer info), #7 logging gaps (CSV, reserved mem, val loss,
  git in records), #10 missing tests (label-shift, tokenizer round-trip).
- **Already resolved (confirm with checks only):** #2 weight init lives in
  `models/transformer.py` (`_init_weights`/`_init_parameters`), not a missing `utils.init_weights`;
  #3 config uses `field(default_factory=...)` (no mutable defaults); #4 `inference.py` uses the clean
  `build_tokenizer`/`generate` API (no `inference=`/`tokenizer=` misuse; hidden globals are confined to
  the offline `tokenizer_lib.functional` pipeline); #5 label shift happens exactly once in the
  dataloader (`x=buf[:-1]`, `y=buf[1:]`) with no second shift in the engine; #8 validation already runs
  under `eval()` + `no_grad`.

**Future: clean package layout (deferred, separately verified)**
Target: move `dataspace/` and `tokenizer_lib/` under `llm_toaster/`, convert top-level `config/`,
`model/`, `utils/` into pure re-export shims (keeping `from config import ...` working), then drop the
generic names from the wheel's discovered packages. This touches many imports/configs and the strict
editable finder, so it gets its own stage with the same import/compile/pytest checks before/after.

## Stage 2 — config / tokenizer / model smoke confirmation (issues #2, #3, #4)

**What changed (tests only — no source changes; these issues were already resolved)**
- `tests/test_config_validation.py`: added `DefaultsRoundTripTests` — proves config sections are *not*
  shared between `ConfigHandler()` instances (the mutable-default bug signature) and that a bare-defaults
  `to_yaml` → `from_yaml` round trip reproduces `to_dict()` **exactly** (every default survives). [#3]
- `tests/test_tokenizers.py` (new): `build_tokenizer(ConfigHandler())` encode→decode round-trips a raw
  prompt, `apply_chat_template` renders + encodes chat text, and the offline `ByteFallbackTokenizer`
  round-trips + prepends EOS. Runs with the real tiktoken-gpt2 backend when available, byte-fallback
  otherwise. [#4]
- `tests/test_legacy_shims.py` (new): the legacy `from model import TransformerModel` positional shim
  instantiates, runs a forward pass on **random** input (shape ok), `from utils import count_parameters`
  reports an `"…M"` string, and init is deterministic under a fixed `torch.manual_seed`. Confirms init
  lives in `models/transformer.py` (`_init_weights`/`_init_parameters`), not a missing `utils.init_weights`. [#2]

**Verified**
- New tests: 9 passed. Full suite: **84 passed** (was 75). `compileall` clean. Tokenizer round-trip
  exercised the real tiktoken-gpt2 backend (vocab 50304).

**Outcome:** issues #2, #3, #4 confirmed already-correct and now locked by tests. No production code changed.

## Stage 3 — training loss / label-shift correctness (issue #5)

**What changed (tests only — labels are already shifted exactly once)**
- `tests/test_data_loader.py`: added `LabelShiftContractTests` — for tokens `[10, 11, 12, 13]` the
  loader yields `x=[[10,11,12]]`, `y=[[11,12,13]]`, i.e. pairs `10->11, 11->12, 12->13`, and the
  single-shift invariant `y[:-1] == x[1:]`. Torch-free.
- `tests/test_engine_components.py`: added `LabelShiftLossTests` — runs the real
  `DataLoaderLite -> build_model -> CrossEntropyLoss` path and confirms a finite, positive loss with
  logits/targets aligned position-for-position. This mirrors `engine.train_step` exactly: the loss is
  computed on `(logits, y)` with **no** second shift (`logits[:, :-1]` / `y[:, 1:]` does not exist).

**Decision (documented):** label shifting happens in exactly ONE place — the dataloader
(`data_loader.py`: `x=buf[:-1]`, `y=buf[1:]`). The trainer compares logits to targets directly. No
two-token-ahead bug exists.

**Verified:** new tests pass; full suite **86 passed**; `compileall` clean.

## Stage 4 — checkpoint / resume reliability (issues #6, #9)

**Production changes**
- `training/checkpointing.py`: new `atomic_save(payload, path)` — writes to a temp file in the same
  directory, `flush` + `os.fsync`, then `os.replace` (atomic). `save_checkpoint` now uses it, so an
  interruption mid-write can never corrupt the live checkpoint and never leaves a partial `.tmp_ckpt_*`
  behind. Payload gains `wall_clock_s` (cumulative elapsed seconds) and `tokenizer_info` (special-token
  ids + vocab) (#9, #6).
- `training/engine.py`:
  - Tracks cumulative elapsed **seconds** across resumes via `perf_counter` deltas (`wall_clock_s`,
    `_elapsed_seconds_total`) — restored from the checkpoint, persisted on save, and reported as the
    `elapsed_s`/`eta_s` metrics. No wall-clock timestamps are used as duration (#6).
  - SIGINT/SIGTERM handling: a handler **requests** a graceful stop (sets a flag); the loop then saves
    a *consistent* emergency checkpoint (`<output_dir>/emergency.pt`) at the next step boundary and
    stops — avoiding a save mid-optimizer-step. A second interrupt restores the default handler and
    force-quits. A `KeyboardInterrupt` caught around the loop also triggers the emergency save. Handlers
    are installed only on the main thread and always restored in `finally` (#9).
  - `save_checkpoint`/`_save_step_checkpoint` pass `wall_clock_s` + `tokenizer_info` through.

**Decisions / notes**
- Duration = elapsed seconds (monotonic deltas), not timestamps — survives resume correctly.
- Emergency save at a step boundary (not inside the signal handler) guarantees a consistent
  model/optimizer/RNG snapshot. Limitation: a single hung step delays the save until it returns.
- `load_checkpoint` already raises clearly on missing/corrupt/newer-format checkpoints (never swallows).

**Verified**
- New tests: atomic save leaves no temp file + carries `wall_clock_s`; missing checkpoint raises
  `FileNotFoundError`; corrupt checkpoint raises; interrupt handler only flags (no save/raise);
  emergency checkpoint saved at boundary and loop stops early; N vs (K + resume) reaches the same final
  step count + tokens with elapsed time persisted.
- Full suite **92 passed**. `ruff check` + `ruff format --check` clean.
- Live end-to-end interrupt: launched a real CPU run, sent SIGINT mid-loop → graceful exit (code 0),
  `emergency.pt` written atomically at step boundary (step 25), `wall_clock_s` recorded, no temp leftovers.

## Stage 5 — structured logging + validation (issues #7, #8)

**Production changes**
- `training/metrics.py`: new `CsvMetricsWriter` — append-only, rectangular CSV of per-step records
  (fixed header from the first record / existing file; extra keys dropped, missing keys blank; flushes
  each row). Complements the existing JSONL writer (#7).
- `config/schema.py`: `LoggingConfig.metrics_csv` (None → derive `metrics.csv` next to `metrics.jsonl`;
  "" disables). Per-run sweep dirs get their own CSV automatically.
- `training/engine.py`:
  - Per-step records now also include `val_loss`, `val_perplexity`, `peak_mem_reserved_bytes`
    (alongside the existing allocated `peak_mem_bytes`), `iter_time_ms` (step time), and `resumed`.
    The architecture row now carries `git_commit`, `config_path`, and `resumed` (#7).
  - Validation is evaluated *before* the log block so the freshest `val_loss` lands in the same record;
    `last_val_loss` is tracked and surfaced. A resolved-config snapshot is written up front (#7, #8).
  - Step records are written to both JSONL and the new CSV; both writers are closed in `finally`.

**Verified**
- New test `test_structured_logging_csv_jsonl_and_validation`: JSONL architecture row has git/config;
  step rows carry the required keys + `peak_mem_reserved_bytes`; validation loss is surfaced; an oversized
  `eval_steps` (more than available val batches) wraps shards without crashing; CSV exists, parses, and
  every row has `step/loss/tokens_per_sec/elapsed_s` populated (#7, #8).
- Inspected real output: arch row keys include `git_commit`/`config_path`/`resumed`; CSV header is
  rectangular (`step,…,val_loss,val_perplexity,iter_time_ms,…,peak_mem_reserved_bytes,resumed`).
- Full suite **93 passed**; `ruff check` + `ruff format --check` clean. Note: `elapsed_s` is the elapsed
  time field (kept its name; aggregate.py is unaffected — it reads `peak_mem_bytes`, still present).

## Stage 6 — minimal tests + final smoke run (issue #10)

**Audit: the minimal-test checklist is fully covered (no new duplicate tests needed)**
- config round-trip → `tests/test_config_validation.py::DefaultsRoundTripTests`
- tokenizer encode/decode → `tests/test_tokenizers.py::TokenizerRoundTripTests`
- dataloader label-shift → `tests/test_data_loader.py::LabelShiftContractTests`
- model forward shape → `tests/test_models_and_lora.py::ModelMatrixTests` (+ `test_engine_components`)
- one-batch training loss → `tests/test_engine_components.py` (`test_train_step_advances_and_is_finite`,
  `LabelShiftLossTests`)
- checkpoint save/load/resume smoke → `tests/test_training_smoke.py` (resume tests),
  `tests/test_engine_components.py::EngineTrainingTests::test_checkpoint_resume_restores_progress`

**Final verification (the CI gate + the requested smoke)**
- `ruff check .` clean; `ruff format --check .` clean (after a pre-existing `experiments/sweep.py`
  reformat, committed separately — a file untouched by these stages).
- `python -m compileall .` clean.
- `pytest --cov --cov-report=term-missing` → **all tests pass, total coverage 86.48% (gate 80%)**.
- **Live end-to-end CPU run via the scripts** (in a temp dir, repo left clean): `python trainer.py
  --config <smoke> --mode pretrain` produced `ckpt`, `metrics.jsonl`, `metrics.csv`, the resolved-config
  snapshot, and `train.log` (with `val_loss` surfaced); `python inference.py --config <smoke> --model
  <ckpt> -p "hello world"` loaded that checkpoint and generated text. Both scripts exited 0.

**Outcome:** all ten reported issues are resolved or confirmed-already-correct, each locked by a test.
`git status` clean; the suite, lint, compile, and an end-to-end CPU train→infer run all pass.

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

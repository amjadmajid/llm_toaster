#!/usr/bin/env python
"""Manifest-backed data tooling: prepare / inspect / validate / migrate-legacy.

Examples:
    python scripts/data.py prepare  --config config/default_config.yaml
    python scripts/data.py prepare  --config config/default_config.yaml --dry-run
    python scripts/data.py inspect  --manifest dataspace/fineweb/manifest.json
    python scripts/data.py validate --manifest dataspace/fineweb/manifest.json
    python scripts/data.py migrate-legacy --data-dir dataspace/fineweb \\
        --manifest dataspace/fineweb/manifest.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_toaster.toaster.data.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

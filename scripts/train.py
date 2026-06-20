#!/usr/bin/env python
"""Train via the modular engine while preserving trainer.py semantics."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trainer import main

if __name__ == "__main__":
    main()

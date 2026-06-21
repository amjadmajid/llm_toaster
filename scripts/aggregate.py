#!/usr/bin/env python
"""Aggregate sweep runs: python scripts/aggregate.py --dir runs/ffn_ablation --csv out.csv --plot"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_toaster.toaster.experiments.aggregate import main

if __name__ == "__main__":
    main()

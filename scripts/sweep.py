#!/usr/bin/env python
"""Run an architecture sweep: python scripts/sweep.py --spec config/sweeps/example.yaml"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_toaster.toaster.experiments.sweep import main

if __name__ == "__main__":
    main()

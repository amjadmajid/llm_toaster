#!/usr/bin/env python
"""Run ON the target device (Jetson / laptop):
    python scripts/bench_device.py --config model/babyGPT/babyGPT_base.yaml \
        --checkpoint checkpoints/base_ckpt --device-name jetson-agx --precision fp16
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_toaster.toaster.experiments.bench import main

if __name__ == "__main__":
    main()

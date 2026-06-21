#!/usr/bin/env python
"""Print/write an architecture card (Markdown + Mermaid + exact numbers) for a config:
python scripts/describe_arch.py --config config/default_config.yaml
python scripts/describe_arch.py --config config/default_config.yaml --out docs/cards/base.md
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_toaster.toaster.config import ConfigHandler
from llm_toaster.toaster.models.registry import build_model
from llm_toaster.toaster.report import architecture_report
from llm_toaster.toaster.tokenizers import build_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an architecture card for a config.")
    parser.add_argument("--config", default="config/default_config.yaml")
    parser.add_argument("--out", default=None, help="Write Markdown here (default: stdout).")
    args = parser.parse_args()

    config = ConfigHandler.from_yaml(args.config)
    if config.model.vocab_size is None:
        config.model.vocab_size = build_tokenizer(config).vocab_size
    report = architecture_report(build_model(config), config)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"Wrote {out}")
    else:
        print(report)


if __name__ == "__main__":
    main()

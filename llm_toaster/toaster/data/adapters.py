"""JSONL data adapters for pretraining, SFT, chat, and preference rows."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class FormattedExample:
    prompt: str
    response: str
    assistant_spans: list[tuple[int, int]] | None = None


@dataclass(frozen=True)
class DataAdapter:
    name: str
    can_handle: Callable[[dict], bool]
    formatter: Callable[[dict, object | None], FormattedExample]


class DataAdapterRegistry:
    """Registry for converting common JSONL schemas into prompt/response text."""

    _items: dict[str, DataAdapter] = {}

    @classmethod
    def register(cls, adapter: DataAdapter) -> DataAdapter:
        cls._items[adapter.name] = adapter
        return adapter

    @classmethod
    def get(cls, name: str) -> DataAdapter:
        if name not in cls._items:
            raise ValueError(f"Unknown data format {name!r}. Available formats: {sorted(cls._items)}")
        return cls._items[name]

    @classmethod
    def detect(cls, row: dict) -> DataAdapter:
        matches = [adapter for adapter in cls._items.values() if adapter.can_handle(row)]
        if not matches:
            raise ValueError(f"Unsupported row schema with keys {sorted(row.keys())}")
        return matches[0]

    @classmethod
    def format_row(cls, row: dict, fmt: str = "auto", tokenizer=None) -> tuple[str, str]:
        example = cls.format_example(row, fmt=fmt, tokenizer=tokenizer)
        return example.prompt, example.response

    @classmethod
    def format_example(cls, row: dict, fmt: str = "auto", tokenizer=None) -> FormattedExample:
        adapter = cls.detect(row) if fmt == "auto" else cls.get(fmt)
        return adapter.formatter(row, tokenizer)


class JsonlSFTDataLoader:
    """Small CPU-friendly SFT loader with prompt masking.

    Rows are formatted through ``DataAdapterRegistry``.  By default, prompt
    tokens are masked to ``-100`` so only assistant/response tokens contribute
    to the language-modeling loss.
    """

    def __init__(
        self,
        B: int,
        T: int,
        path: str,
        tokenizer,
        fmt: str = "auto",
        train_on_prompt: bool = False,
        seed: int = 1337,
        shuffle: bool = True,
    ):
        if B <= 0 or T <= 0:
            raise ValueError("Batch size and sequence length must be positive")
        dataset_path = Path(path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"SFT dataset does not exist: {dataset_path}")

        self.B = B
        self.T = T
        self.tokenizer = tokenizer
        self.fmt = fmt
        self.train_on_prompt = train_on_prompt
        self.rng = random.Random(seed)
        self.shuffle = shuffle
        self.index = 0
        self.examples = self._load_examples(dataset_path)
        if not self.examples:
            raise ValueError(f"No SFT examples found in {dataset_path}")
        if self.shuffle:
            self.rng.shuffle(self.examples)

    def _load_examples(self, path: Path) -> list[tuple[list[int], list[int] | None]]:
        examples = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    examples.append(self.encode_row(row))
                except Exception as exc:
                    raise ValueError(f"Failed to parse {path}:{line_no}: {exc}") from exc
        return examples

    def encode_row(self, row: dict) -> tuple[list[int], list[int] | None]:
        example = DataAdapterRegistry.format_example(row, fmt=self.fmt, tokenizer=self.tokenizer)
        if example.response == "":
            ids = self._encode(example.prompt) + [self._eos_id()]
            return ids[: self.T + 1], None

        prompt_ids = self._encode(example.prompt)
        response_ids = self._encode(example.response) + [self._eos_id()]
        prompt_ids, response_ids = self._truncate_prompt_first(prompt_ids, response_ids)
        ids = (prompt_ids + response_ids)[: self.T + 1]
        labels = ids[1:] + [-100]
        if not self.train_on_prompt:
            mask_count = max(0, min(len(prompt_ids) - 1, len(labels)))
            labels[:mask_count] = [-100] * mask_count
        return ids, labels

    def _truncate_prompt_first(self, prompt_ids: list[int], response_ids: list[int]) -> tuple[list[int], list[int]]:
        if len(prompt_ids) + len(response_ids) <= self.T + 1:
            return prompt_ids, response_ids
        max_prompt = max(0, self.T + 1 - len(response_ids))
        if max_prompt:
            prompt_ids = prompt_ids[-max_prompt:]
        else:
            prompt_ids = []
            response_ids = response_ids[: self.T + 1]
        return prompt_ids, response_ids

    def _encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text))

    def _eos_id(self) -> int:
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        return 0 if eos_id is None else int(eos_id)

    def next_batch(self):
        x = np.zeros((self.B, self.T), dtype=np.int64)
        y = np.full((self.B, self.T), -100, dtype=np.int64)
        for batch_index in range(self.B):
            ids, labels = self._next_example()
            if len(ids) < 2:
                continue
            source = ids[:-1][: self.T]
            target = labels[: self.T] if labels is not None else ids[1:][: self.T]
            x[batch_index, : len(source)] = source
            y[batch_index, : len(target)] = target
        return x, y, 0

    def state_dict(self) -> dict:
        return {"index": self.index}

    def load_state_dict(self, state: dict) -> None:
        self.index = int(state.get("index", 0)) % len(self.examples)

    def _next_example(self) -> tuple[list[int], list[int] | None]:
        example = self.examples[self.index]
        self.index += 1
        if self.index >= len(self.examples):
            self.index = 0
            if self.shuffle:
                self.rng.shuffle(self.examples)
        return example


def _text(row: dict, _tok=None) -> FormattedExample:
    return FormattedExample(prompt=row["text"], response="")


def _prompt_completion(row: dict, _tok=None) -> FormattedExample:
    return FormattedExample(prompt=row["prompt"], response=row["completion"])


def _instruction_response(row: dict, _tok=None) -> FormattedExample:
    return FormattedExample(prompt=f"Instruction:\n{row['instruction']}\n\nResponse:\n", response=row["response"])


def _alpaca(row: dict, _tok=None) -> FormattedExample:
    input_text = row.get("input", "")
    prompt = f"Instruction:\n{row['instruction']}\n"
    if input_text:
        prompt += f"Input:\n{input_text}\n"
    prompt += "\nResponse:\n"
    return FormattedExample(prompt=prompt, response=row["output"])


def _openai_messages(row: dict, tokenizer=None) -> FormattedExample:
    messages = row["messages"]
    prompt_messages = []
    answer = ""
    for message in messages:
        if message.get("role") == "assistant":
            answer = message.get("content", "")
        else:
            prompt_messages.append(message)
    prompt = tokenizer.apply_chat_template(prompt_messages) if tokenizer else _simple_chat_template(prompt_messages)
    return FormattedExample(prompt=prompt, response=answer)


def _sharegpt(row: dict, _tok=None) -> FormattedExample:
    prompt = ""
    answer = ""
    for message in row["conversations"]:
        sender = message.get("from")
        value = message.get("value", "")
        if sender in {"gpt", "assistant"}:
            answer = value
        else:
            prompt += f"user: {value}\nassistant: "
    return FormattedExample(prompt=prompt, response=answer)


def _preference_dpo(row: dict, _tok=None) -> FormattedExample:
    # The SFT loader trains on the chosen answer. A future DPO trainer can use
    # the same registry and consume rejected responses explicitly.
    return FormattedExample(prompt=row["prompt"], response=row["chosen"])


def _simple_chat_template(messages: list[dict]) -> str:
    return (
        "".join(f"{message.get('role', 'user')}: {message.get('content', '')}\n" for message in messages)
        + "assistant: "
    )


DataAdapterRegistry.register(DataAdapter("text", lambda row: "text" in row, _text))
DataAdapterRegistry.register(
    DataAdapter("prompt_completion", lambda row: {"prompt", "completion"}.issubset(row), _prompt_completion)
)
DataAdapterRegistry.register(
    DataAdapter("instruction_response", lambda row: {"instruction", "response"}.issubset(row), _instruction_response)
)
DataAdapterRegistry.register(DataAdapter("alpaca", lambda row: {"instruction", "output"}.issubset(row), _alpaca))
DataAdapterRegistry.register(DataAdapter("openai_messages", lambda row: "messages" in row, _openai_messages))
DataAdapterRegistry.register(DataAdapter("sharegpt", lambda row: "conversations" in row, _sharegpt))
DataAdapterRegistry.register(
    DataAdapter("preference_dpo", lambda row: {"prompt", "chosen", "rejected"}.issubset(row), _preference_dpo)
)

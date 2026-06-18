"""Tokenizer abstractions used by the training engine."""

from __future__ import annotations


class BaseTokenizer:
    """Small common tokenizer interface used by loaders and scripts."""

    bos_token_id: int | None = None
    eos_token_id: int | None = 50256
    pad_token_id: int | None = 0
    vocab_size: int = 50304

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError

    def apply_chat_template(self, messages: list[dict]) -> str:
        lines = [f"{message.get('role', 'user')}: {message.get('content', '')}" for message in messages]
        return "\n".join(lines) + "\nassistant: "


class ByteFallbackTokenizer(BaseTokenizer):
    """Offline byte-level fallback used when tiktoken assets are unavailable."""

    eos_token_id = 0
    pad_token_id = 0
    vocab_size = 50304

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [byte + 1 for byte in text.encode("utf-8")]
        if add_special_tokens:
            return [self.eos_token_id] + ids
        return ids

    def decode(self, ids: list[int]) -> str:
        values = [max(0, min(255, int(token) - 1)) for token in ids if int(token) > 0]
        return bytes(values).decode("utf-8", errors="ignore")


class TiktokenTokenizer(BaseTokenizer):
    """GPT-style tiktoken tokenizer with an offline byte fallback."""

    def __init__(self, name: str = "gpt2", allow_fallback: bool = True):
        self._fallback: ByteFallbackTokenizer | None = None
        try:
            import tiktoken

            self.encoding = tiktoken.get_encoding(name)
            self.eos_token_id = self.encoding.eot_token
            self.pad_token_id = 0
            self.vocab_size = 50304
        except Exception as exc:
            if not allow_fallback:
                raise RuntimeError(f"Unable to initialize tiktoken encoding {name!r}") from exc
            self.encoding = None
            self._fallback = ByteFallbackTokenizer()
            self.eos_token_id = self._fallback.eos_token_id
            self.pad_token_id = self._fallback.pad_token_id
            self.vocab_size = self._fallback.vocab_size

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if self._fallback is not None:
            return self._fallback.encode(text, add_special_tokens=add_special_tokens)
        ids = self.encoding.encode(text)
        if add_special_tokens:
            return [self.eos_token_id] + ids
        return ids

    def decode(self, ids: list[int]) -> str:
        if self._fallback is not None:
            return self._fallback.decode(ids)
        return self.encoding.decode(list(ids))


class HFTokenizer(BaseTokenizer):
    """Hugging Face tokenizer wrapper loaded lazily."""

    def __init__(self, name_or_path: str = "gpt2"):
        try:
            from transformers import AutoTokenizer
        except ModuleNotFoundError as exc:
            raise ImportError("Install transformers to use tokenizer.type='hf'.") from exc

        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        self.vocab_size = len(self.tokenizer)
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id or self.eos_token_id or 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def apply_chat_template(self, messages: list[dict]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False)
        return super().apply_chat_template(messages)


class SentencePieceTokenizer(BaseTokenizer):
    """Reserved SentencePiece integration point."""

    def __init__(self, *_args, **_kwargs):
        raise ImportError("SentencePieceTokenizer is a stub; install/provide sentencepiece integration first.")


def build_tokenizer(config) -> BaseTokenizer:
    tokenizer_type = getattr(config.tokenizer, "type", "tiktoken").lower()
    if tokenizer_type in {"gpt2", "tiktoken"}:
        return TiktokenTokenizer(getattr(config.tokenizer, "name", "gpt2"))
    if tokenizer_type in {"hf", "huggingface"}:
        return HFTokenizer(config.tokenizer.path or config.tokenizer.name)
    if tokenizer_type in {"sp", "sentencepiece"}:
        return SentencePieceTokenizer(config.tokenizer.path)
    raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

"""Issue #4 confirmation: the tokenizer API is clean and predictable.

``build_tokenizer`` returns a tokenizer whose ``encode``/``decode`` round-trip and whose
``apply_chat_template`` renders a chat string. These run offline: ``TiktokenTokenizer`` falls
back to a byte-level encoder when tiktoken assets are unavailable, and both backends round-trip
plain ASCII text exactly.
"""

import unittest

from llm_toaster.toaster.config import ConfigHandler
from llm_toaster.toaster.tokenizers import ByteFallbackTokenizer, build_tokenizer


class TokenizerRoundTripTests(unittest.TestCase):
    def test_build_tokenizer_encodes_and_decodes_raw_prompt(self):
        tokenizer = build_tokenizer(ConfigHandler())
        text = "Hello, world! This is a tokenizer smoke test."
        ids = tokenizer.encode(text)
        self.assertGreater(len(ids), 0)
        self.assertTrue(all(isinstance(i, int) for i in ids))
        self.assertEqual(tokenizer.decode(ids), text)

    def test_chat_format_text_encodes(self):
        tokenizer = build_tokenizer(ConfigHandler())
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        )
        self.assertIn("user", rendered)
        self.assertIn("assistant", rendered)
        self.assertGreater(len(tokenizer.encode(rendered)), 0)

    def test_byte_fallback_roundtrips_offline(self):
        tokenizer = ByteFallbackTokenizer()
        text = "offline byte fallback 123"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_add_special_tokens_prepends_eos(self):
        tokenizer = ByteFallbackTokenizer()
        plain = tokenizer.encode("x")
        with_special = tokenizer.encode("x", add_special_tokens=True)
        self.assertEqual(with_special, [tokenizer.eos_token_id, *plain])


if __name__ == "__main__":
    unittest.main()

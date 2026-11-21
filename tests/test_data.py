import unittest

from safety_project_a4.data import encode_pair


class DummyTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.assert_false = tokenize
        prompt = f"<user>{messages[0]['content']}</user><assistant>"
        if add_generation_prompt:
            return prompt
        return prompt + messages[1]["content"] + "</assistant>"

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [[ord(character) for character in text]]}


class DataEncodingTests(unittest.TestCase):
    def test_assistant_only_mask_and_nested_tokenizer_output(self):
        encoded = encode_pair(DummyTokenizer(), "2+2?", "4", max_length=128)
        first_label = next(index for index, label in enumerate(encoded.labels) if label != -100)
        self.assertEqual(encoded.labels[:first_label], [-100] * first_label)
        self.assertEqual(encoded.labels[first_label:], encoded.input_ids[first_label:])

    def test_rejects_truncation(self):
        with self.assertRaisesRegex(ValueError, "increase the configured limit instead of truncating"):
            encode_pair(DummyTokenizer(), "long question", "long answer", max_length=8)


if __name__ == "__main__":
    unittest.main()

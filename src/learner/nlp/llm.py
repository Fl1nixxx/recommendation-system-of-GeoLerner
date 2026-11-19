from __future__ import annotations

import json
import torch
from typing import Any


class QwenResponseError(RuntimeError):

class QwenJSONClient:
    def __init__(self,model: Any,tokenizer: Any,*,max_new_tokens: int = 700,retries: int = 1) -> None:

        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if retries < 0:
            raise ValueError("retries cannot be negative")

        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.retries = retries

    def _input_device(self):
        device = getattr(self.model, "device", None)
        if device is not None:
            return device
        try:
            return next(self.model.parameters()).device
        except (AttributeError, StopIteration):
            return None

    @staticmethod
    def _decode_json(raw: str) -> dict[str, Any]:
        start = raw.find("{")
        if start < 0:
            raise ValueError("JSON object not found in model response")
        parsed, _ = json.JSONDecoder().raw_decode(raw[start:])
        if not isinstance(parsed, dict):
            raise ValueError("Expected a JSON object")
        return parsed

    def _generate(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}]

        formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(formatted, return_tensors="pt")
        device = self._input_device()
        if device is not None:
            inputs = inputs.to(device)

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False}

        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

        if pad_token_id is None:
            pad_token_id = eos_token_id
        if pad_token_id is not None:
            generation_kwargs["pad_token_id"] = pad_token_id

        with torch.inference_mode():
            output = self.model.generate(**inputs, **generation_kwargs)

        sequences = getattr(output, "sequences", output)
        prompt_length = inputs["input_ids"].shape[1]
        generated_tokens = sequences[0][prompt_length:]

        return self.tokenizer.decode(generated_tokens,skip_special_tokens=True).strip()

    def ask_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        last_error: Exception | None = None
        prompt = user_prompt

        for attempt in range(self.retries + 1):
            raw = self._generate(system_prompt, prompt)
            try:
                return self._decode_json(raw)
            except (ValueError, json.JSONDecodeError) as error:
                last_error = error
                prompt = (
                    user_prompt
                    + "\n\nYour previous response was not valid JSON. "
                    "Return exactly one valid JSON object, without Markdown.")

        raise QwenResponseError(f"Qwen failed to return valid JSON: {last_error}")

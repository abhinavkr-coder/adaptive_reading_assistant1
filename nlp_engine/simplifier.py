"""Lazy-loads the fine-tuned T5 model and runs inference."""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

_DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "models", "t5-simplifier")


class Simplifier:
    def __init__(self, model_path: str = _DEFAULT_MODEL):
        self._path  = model_path
        self._ready = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load(self):
        if self._ready:
            return

        if not os.path.isdir(self._path):
            print(f"[Simplifier] WARNING: model not found at {self._path}. "
                "Train and copy the model first. Returning source unchanged.")
            self._ready = True
            self._missing = True
            return

        print(f"[Simplifier] Loading from {self._path}…")
        self.tokenizer = AutoTokenizer.from_pretrained(self._path)

        # Avoid meta-tensor loading path that breaks .to()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self._path,
            low_cpu_mem_usage=False
        ).to(self.device)

        self.model.eval()
        self._missing = False
        self._ready = True
        print("[Simplifier] Ready.")

    def simplify(self, text: str, max_new_tokens: int = 200) -> str:
        self._load()
        if getattr(self, "_missing", False):
            return text   # graceful fallback when model isn't trained yet
        inputs = self.tokenizer(
            f"simplify: {text}",
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                      num_beams=4, early_stopping=True)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)
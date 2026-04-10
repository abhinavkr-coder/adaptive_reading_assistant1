"""
Loads a T5-family model for sentence simplification.

Priority order:
  1. Your custom fine-tuned model at nlp_engine/models/t5-simplifier/
     (drop it here after Colab training — the code picks it up automatically)
  2. google/flan-t5-base as a reliable pre-trained fallback.
     This model is instruction-tuned, so it responds well to natural-language
     prompts like "Simplify this sentence for a beginner: ..."

Why flan-t5-base instead of the old sander-wood model?
  The sander-wood repository has been removed from HuggingFace (returns 401).
  flan-t5-base is permanently maintained by Google, is publicly available
  without authentication, and produces clean simplified output.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Path where you'll place your Colab-trained model once it's ready
_CUSTOM_MODEL = os.path.join(os.path.dirname(__file__), "models", "t5-simplifier")

# Reliable public fallback — instruction-tuned T5, great at following prompts
_FALLBACK_MODEL = "google/flan-t5-base"


class Simplifier:
    def __init__(self):
        if os.path.isdir(_CUSTOM_MODEL):
            self._path  = _CUSTOM_MODEL
            self._label = "custom fine-tuned"
        else:
            self._path  = _FALLBACK_MODEL
            self._label = f"pre-trained fallback ({_FALLBACK_MODEL})"

        self._ready = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load(self):
        if self._ready:
            return

        print(f"[Simplifier] Loading {self._label} from: {self._path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self._path)

        # low_cpu_mem_usage=False avoids the "meta tensor" crash in newer
        # versions of transformers when loading on CPU.
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self._path,
            low_cpu_mem_usage=False,
        ).to(self.device)

        self.model.eval()
        self._ready = True
        print(f"[Simplifier] Ready on {self.device}.")

    def _build_prompt(self, text: str) -> str:
        """
        For your custom fine-tuned model the prompt is 'simplify: ...',
        which matches the training format from data_builder.py.

        For flan-t5-base we use a more descriptive instruction because
        the model was trained to follow natural language directives —
        the richer the instruction, the better the output.
        """
        if self._path == _FALLBACK_MODEL:
            return (
                f"Simplify the following sentence so that a beginner "
                f"can understand it easily. Use short words and short sentences. "
                f"Sentence: {text}"
            )
        # Custom model expects the same prefix used during training
        return f"simplify: {text}"

    def simplify(self, text: str) -> str:
        self._load()

        prompt = self._build_prompt(text)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=200,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,   # prevents repeating any 3-word phrase
                repetition_penalty=2.5,   # penalises tokens already generated
                length_penalty=1.0,       # neutral — don't reward/punish length
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)
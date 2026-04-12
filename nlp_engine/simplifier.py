"""
Paragraph-level vocabulary simplifier using google/flan-t5-large.

THE CORE PROMPT ENGINEERING PROBLEM — AND HOW WE SOLVE IT
──────────────────────────────────────────────────────────
The original prompts caused flan-T5 to summarise paragraphs, losing content.
This happened because "simplify" is ambiguous: to a language model trained on
summarisation and simplification tasks alike, it can mean "make shorter."

The fix is a structured, sentence-counted prompt that makes three things
explicit to the model:

  1. COUNT CONSTRAINT  — the output must contain exactly N sentences,
     where N is counted from the input. The model is unlikely to drop a
     sentence when the prompt names the exact number expected.

  2. NEGATIVE RULES    — "Do NOT summarise. Do NOT omit any sentence."
     Explicit negatives are highly effective for instruction-tuned models.

  3. TASK FOCUS        — the only permitted change is word choice. Structure,
     facts, names, and numbers must be identical in the output.

QUALITY GATE
────────────
Even with good prompts, models occasionally produce shorter output. The
simplify() method compares output length to input length and falls back to
the original text if the model discarded more than 40% of the characters.
This prevents silent information loss from reaching the user.
"""

import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

_CUSTOM_MODEL   = os.path.join(os.path.dirname(__file__), "models", "t5-simplifier")
_FALLBACK_MODEL = "google/flan-t5-large"


def _count_sentences(text: str) -> int:
    """Count sentences using the same regex as ComplexityAnalyzer.split_sentences."""
    return len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()])


class Simplifier:
    def __init__(self):
        if os.path.isdir(_CUSTOM_MODEL):
            self._path      = _CUSTOM_MODEL
            self._is_custom = True
            self._label     = "custom fine-tuned"
        else:
            self._path      = _FALLBACK_MODEL
            self._is_custom = False
            self._label     = _FALLBACK_MODEL

        self._ready = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load(self):
        if self._ready:
            return
        print(f"[Simplifier] Loading '{self._label}' on {self.device} …")
        self.tokenizer = AutoTokenizer.from_pretrained(self._path)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self._path,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,  # prevents the "meta tensor" crash
        ).to(self.device)
        self.model.eval()
        self._ready = True
        print(f"[Simplifier] Ready ({dtype}).")

    def _build_prompt(self, text: str, user_level: str) -> str:
        """
        Build a prompt that instructs flan-T5 to lower the vocabulary level
        WITHOUT removing or summarising any content.

        The sentence count is computed from the input and stated explicitly
        in the prompt.  This is the single most effective technique for
        preventing the model from dropping sentences.
        """
        if self._is_custom:
            # Custom model was trained with this exact prefix
            return f"simplify: {text}"

        n = _count_sentences(text)
        sentence_word = "sentence" if n == 1 else "sentences"

        level_instruction = {
            "beginner": (
                "Replace every rare or difficult word with a simple, everyday word. "
                "Use short sentences. Aim for a reading level suitable for a 10-year-old."
            ),
            "intermediate": (
                "Replace complex or formal vocabulary with clearer alternatives. "
                "Break very long sentences into shorter ones where needed. "
                "Aim for a high-school reading level."
            ),
            "advanced": (
                "Lightly improve clarity. Replace only the most convoluted phrases "
                "with cleaner alternatives. Preserve all technical vocabulary and nuance. "
                "Aim for a university reading level."
            ),
        }.get(user_level, "Replace difficult words with simpler ones.")

        return (
            f"Your task is to rewrite the following text using simpler English vocabulary.\n\n"
            f"Instruction: {level_instruction}\n\n"
            f"STRICT RULES — you must follow all of these:\n"
            f"  1. The input has {n} {sentence_word}. Your output must also have "
            f"exactly {n} {sentence_word}. Do not delete, merge, or skip any.\n"
            f"  2. Keep every fact, name, number, date, and event exactly as stated.\n"
            f"  3. Only change vocabulary. Do NOT summarise, shorten, or condense the text.\n"
            f"  4. Do NOT add new information that was not in the original.\n\n"
            f"Text to rewrite ({n} {sentence_word}):\n{text}\n\n"
            f"Rewritten text (same {n} {sentence_word}, simpler words only):"
        )

    def simplify(self, text: str, user_level: str = "beginner") -> str:
        self._load()

        prompt = self._build_prompt(text, user_level)

        print(f"\n{'─'*60}")
        print(f"[T5 PROMPT → model]\n{prompt}")
        print(f"{'─'*60}")

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=768,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=400,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                repetition_penalty=1.4,
                length_penalty=1.2,
            )

        result = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        print(f"[T5 OUTPUT ← model]\n{result}")
        print(f"{'─'*60}\n")

        # ── Quality gate ────────────────────────────────────────────────────
        # If the model output is less than 50% the length of the input, it
        # almost certainly summarised rather than simplified.  Return the
        # original so the user never silently loses information.
        if result and len(result) >= len(text) * 0.5:
            return result

        print(
            f"[Simplifier] WARNING: output too short "
            f"({len(result)} chars vs {len(text)} input). "
            f"Returning original to prevent content loss."
        )
        return text
"""
Paragraph-level text rewriter using google/flan-t5-large.

WHY PARAGRAPH-LEVEL?
  Processing individual sentences loses discourse context: pronoun references
  ("it", "they"), topic continuity, and the logical flow between ideas.
  By sending whole paragraphs, the model can maintain coherence across sentences
  and produce rewrites that read naturally rather than as a list of disconnected
  simplified fragments.

WHY flan-t5-large?
  The 'sander-wood' model is gone from HuggingFace (401). flan-t5-large is:
    - Permanently maintained by Google
    - Instruction-tuned on thousands of tasks — including paraphrase and
      text simplification — so it follows level-aware prompts reliably
    - 770M params: heavyweight enough for quality, lightweight enough to load
      on a 4GB+ GPU in fp16 in ~3 seconds

GPU USAGE:
  Install PyTorch with CUDA 12.6 BEFORE running:
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
  The model auto-detects CUDA and uses fp16 on GPU for maximum speed.
  On a mid-range GPU (e.g. RTX 3060), inference per paragraph is ~0.3–0.8s.
  On CPU it falls back to fp32 and will be slower (~8–15s per paragraph).
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ── Model paths ───────────────────────────────────────────────────────────────
# Priority 1: your Colab-trained custom model (place it here after training)
_CUSTOM_MODEL = os.path.join(os.path.dirname(__file__), "models", "t5-simplifier")
# Priority 2: reliable public fallback — instruction-tuned, no authentication needed
_FALLBACK_MODEL = "google/flan-t5-large"

# ── Level-aware prompt templates ──────────────────────────────────────────────
# The more concrete and specific the instruction to flan-t5, the better it follows.
# Each level targets a different CEFR register.
_PROMPTS = {
    "beginner": (
        "Rewrite the following paragraph so a 10-year-old child can understand it. "
        "Rules: use only very simple and common English words; keep every sentence "
        "under 15 words; replace any jargon or technical term with a plain explanation; "
        "preserve every important fact — do not add anything new.\n\n"
        "Paragraph: {text}\n\n"
        "Simplified paragraph:"
    ),
    "intermediate": (
        "Rewrite the following paragraph for a high-school student. "
        "Rules: replace difficult vocabulary with clearer words but keep important "
        "subject-specific terms; break very long sentences into shorter ones; "
        "keep the full meaning and all details; maintain a natural, readable tone.\n\n"
        "Paragraph: {text}\n\n"
        "Rewritten paragraph:"
    ),
    "advanced": (
        "Lightly edit the following paragraph to improve clarity for a well-educated "
        "adult reader. Rules: preserve all technical vocabulary and nuance; improve "
        "sentence flow only where the original is genuinely awkward; do not "
        "oversimplify — this reader welcomes complexity.\n\n"
        "Paragraph: {text}\n\n"
        "Edited paragraph:"
    ),
}

# The custom fine-tuned model uses the same prefix as its training data
_CUSTOM_PROMPT = "simplify: {text}"


class Simplifier:
    def __init__(self):
        # Select model: custom (if trained) or public fallback
        if os.path.isdir(_CUSTOM_MODEL):
            self._path = _CUSTOM_MODEL
            self._is_custom = True
            self._label = "custom fine-tuned"
        else:
            self._path = _FALLBACK_MODEL
            self._is_custom = False
            self._label = _FALLBACK_MODEL

        self._ready = False
        # Use CUDA if available — install torch with cu126 for GPU support
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load(self):
        """Lazy-load the model on the first inference call."""
        if self._ready:
            return

        print(f"[Simplifier] Loading {self._label} on {self.device} …")

        self.tokenizer = AutoTokenizer.from_pretrained(self._path)

        # Use fp16 on CUDA for ~2x speed and half the VRAM.
        # fp32 on CPU because fp16 is poorly supported on CPU in PyTorch.
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self._path,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,  # avoids the meta-tensor copy crash
        ).to(self.device)

        self.model.eval()
        self._ready = True
        print(f"[Simplifier] Ready — {self._label} on {self.device} "
              f"({'fp16' if dtype == torch.float16 else 'fp32'}).")

    def _build_prompt(self, text: str, user_level: str) -> str:
        """Build the prompt. Custom model gets its training prefix; fallback gets
        a rich natural-language instruction that flan-t5 was trained to follow."""
        if self._is_custom:
            return _CUSTOM_PROMPT.format(text=text)
        template = _PROMPTS.get(user_level, _PROMPTS["beginner"])
        return template.format(text=text)

    def simplify(self, text: str, user_level: str = "beginner") -> str:
        """
        Rewrite a paragraph at the given reading level.

        Args:
            text:       The paragraph to simplify (ideally 1–6 sentences).
            user_level: One of "beginner", "intermediate", "advanced".

        Returns:
            The rewritten paragraph. Falls back to the original if the model
            produces an empty string (safeguard against rare generation failures).
        """
        self._load()

        prompt = self._build_prompt(text, user_level)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,   # flan-t5-large max input
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=300,       # enough for a full paragraph
                num_beams=4,              # beam search for coherent output
                early_stopping=True,
                no_repeat_ngram_size=3,   # prevents "the the the" repetition
                repetition_penalty=1.5,   # discourages rewording the same phrase
                length_penalty=1.2,       # slight preference for complete sentences
            )

        result = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        # Safety guard: never return an empty string
        return result if result else text
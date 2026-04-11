"""
Word-level simplification: WordNet candidate retrieval + flan-t5-base selection.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  HOW THIS MAPS TO THE TORCHTEXT T5 TUTORIAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  The tutorial (T5-Base for Summarization, Sentiment & Translation)
  demonstrates that T5 understands tasks purely through PREFIX STRINGS:

    Tutorial task           Prefix used
    ─────────────────────   ─────────────────────────────────────────
    Summarization           "summarize: <article>"
    Sentiment classification "sst2 sentence: <review>"
    EN→DE translation       "translate English to German: <text>"

  We apply the exact same principle for word-level simplification:

    Our task                Prefix / prompt used
    ─────────────────────   ─────────────────────────────────────────
    Candidate selection     "Sentence: "..." The word "X" is difficult.
                             Choose the best simpler replacement from:
                             word1, word2, word3. Reply with one word."
    Direct generation       "Sentence: "..." Replace the difficult verb
                             "X" with a simpler verb. Reply with one word."

  WHY flan-t5-base INSTEAD OF torchtext's T5_BASE_GENERATION:
    torchtext's T5_BASE_GENERATION is standard T5-base pretrained on the
    C4 corpus. It responds to prefixes it was fine-tuned on (summarize,
    translate, etc.) but not to novel natural-language instructions.
    flan-t5-base is the same T5-base architecture BUT further fine-tuned
    by Google on 1,800+ tasks described in natural language — which is
    exactly what our word-replacement prompts are.

  TOKENIZER EQUIVALENCE:
    torchtext tutorial:  T5Transform (SentencePiece, max_seq_len=512)
    Our implementation:  AutoTokenizer.from_pretrained("google/flan-t5-base")
    Both use the same underlying SentencePiece model.

  GENERATION EQUIVALENCE:
    torchtext tutorial:  GenerationUtils(model).generate(input, num_beams=4)
    Our implementation:  model.generate(**inputs, num_beams=4, ...)
    Both perform beam search decoding. We additionally set
    no_repeat_ngram_size=2 to prevent the looping outputs seen earlier.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import json
import logging
import torch
import nltk
from nltk.corpus import wordnet as wn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

nltk.download("wordnet",  quiet=True)
nltk.download("omw-1.4", quiet=True)

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger("ARA.simplifier")

# ── Model paths ───────────────────────────────────────────────────────────────
_CUSTOM_MODEL   = os.path.join(os.path.dirname(__file__), "models", "t5-simplifier")
_FALLBACK_MODEL = "google/flan-t5-base"

# ── WordNet POS mapping ───────────────────────────────────────────────────────
_WN_POS = {"NOUN": wn.NOUN, "VERB": wn.VERB, "ADJ": wn.ADJ, "ADV": wn.ADV}

# ── CEFR numeric rank (lower = easier) ────────────────────────────────────────
_CEFR_RANK = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}

# Maximum hard words processed per sentence — keeps per-sentence latency
# predictable on CPU (each T5 call takes ~300–800 ms on CPU).
_MAX_WORDS_PER_SENTENCE = 5


class Simplifier:
    def __init__(self, cefr_vocab: dict, common_words: set):
        """
        Accepts already-loaded vocabulary maps from ComplexityAnalyzer so
        the CSVs are never parsed twice.
        """
        self.cefr_vocab   = cefr_vocab
        self.common_words = common_words

        # Decide which model to use
        if os.path.isdir(_CUSTOM_MODEL):
            self._path  = _CUSTOM_MODEL
            self._label = "custom fine-tuned model"
        else:
            self._path  = _FALLBACK_MODEL
            self._label = f"pre-trained fallback: {_FALLBACK_MODEL}"

        self._ready  = False
        self._cache: dict[tuple, str] = {}   # (lemma, pos) → replacement
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Model loading (lazy — only triggered on first simplify call) ──────────

    def _load(self):
        if self._ready:
            return
        logger.info(f"Loading {self._label} from: {self._path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self._path)
        # low_cpu_mem_usage=False prevents the "Cannot copy out of meta tensor"
        # crash introduced in transformers ≥ 4.38 when loading on CPU.
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self._path,
            low_cpu_mem_usage=False,
        ).to(self.device)
        self.model.eval()
        self._ready = True
        logger.info(f"Model ready on {self.device}.")

    # ── WordNet helpers ───────────────────────────────────────────────────────

    def _is_simpler_than(self, candidate: str, original: str) -> bool:
        """True when the candidate is easier than the original word."""
        orig_rank = _CEFR_RANK.get(self.cefr_vocab.get(original.lower(), ""), 7)
        cand_rank = _CEFR_RANK.get(self.cefr_vocab.get(candidate.lower(), ""), 7)
        return (candidate.lower() in self.common_words) or (cand_rank < orig_rank)

    def _wordnet_candidates(self, lemma: str, pos: str) -> list[str]:
        """
        Retrieves simpler synonyms from WordNet for the given lemma + POS.
        Sorted so that high-frequency everyday words come first.
        """
        wn_pos = _WN_POS.get(pos)
        if not wn_pos:
            return []

        seen: set[str] = set()
        candidates: list[str] = []

        for synset in wn.synsets(lemma, pos=wn_pos):
            for lem in synset.lemmas():
                word = lem.name().replace("_", " ")
                if (
                    " " not in word               # single word only
                    and word.isalpha()
                    and 2 < len(word) < 14
                    and word.lower() != lemma.lower()
                    and word.lower() not in seen
                    and self._is_simpler_than(word, lemma)
                ):
                    seen.add(word.lower())
                    candidates.append(word)

        # Prefer words that are in our everyday high-frequency list,
        # then sort by length (shorter words are usually simpler).
        candidates.sort(key=lambda w: (w.lower() not in self.common_words, len(w)))
        return candidates[:8]

    # ── flan-T5 helpers ───────────────────────────────────────────────────────

    def _generate(self, prompt: str, max_new_tokens: int = 8) -> str:
        """
        Core generation call — equivalent to torchtext's:
            GenerationUtils(model).generate(transform(prompt), num_beams=4)

        We log the full prompt and the raw output so they appear in the
        terminal exactly as the user requested.
        """
        self._load()

        # ── TERMINAL LOG: prompt going INTO the model ──────────────────────
        logger.info(
            "\n"
            "┌─ FLAN-T5 INPUT PROMPT " + "─" * 55 + "\n"
            f"│  {prompt}\n"
            "└" + "─" * 78
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,          # same beam width as the torchtext tutorial
                early_stopping=True,
                no_repeat_ngram_size=2,
            )

        result = self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

        # ── TERMINAL LOG: raw output FROM the model ────────────────────────
        logger.info(
            "\n"
            "┌─ FLAN-T5 RAW OUTPUT " + "─" * 57 + "\n"
            f"│  {result!r}\n"
            "└" + "─" * 78
        )
        return result

    def _t5_pick_from_candidates(
        self, sentence: str, word: str, candidates: list[str]
    ) -> str:
        """
        Asks flan-T5 to choose the most natural-sounding replacement from
        the WordNet candidate list given the full sentence context.

        This mirrors the torchtext tutorial's task-prefix approach:
        instead of "sst2 sentence: <text>" for sentiment, we use a natural
        language instruction that describes our simplification task.
        """
        candidate_str = ", ".join(candidates[:6])
        prompt = (
            f'Sentence: "{sentence}"\n'
            f'The word "{word}" may be difficult to understand. '
            f'Choose the single best, simpler replacement from this list: {candidate_str}. '
            f'Reply with only that one word and nothing else.'
        )
        result = self._generate(prompt, max_new_tokens=8)

        # Validate: the result must be one of the provided candidates
        first = result.split()[0].strip(".,;:\"'").lower() if result.split() else ""
        for c in candidates:
            if c.lower() == first:
                logger.debug(f"Candidate selected: '{word}' → '{c}'")
                return c

        # If T5's response wasn't a valid candidate, fall back to the
        # most common word in the list (index 0, already sorted by frequency).
        logger.debug(
            f"T5 output '{first}' not in candidates — using fallback: '{candidates[0]}'"
        )
        return candidates[0]

    def _t5_generate_replacement(
        self, sentence: str, word: str, pos: str
    ) -> str:
        """
        When WordNet has no simpler synonyms, asks flan-T5 to generate
        a replacement directly. We constrain max_new_tokens=8 so the model
        can only output a single word, preventing the looping seen earlier.
        """
        pos_label = {
            "NOUN": "noun", "VERB": "verb",
            "ADJ": "adjective", "ADV": "adverb"
        }.get(pos, "word")

        prompt = (
            f'Sentence: "{sentence}"\n'
            f'The {pos_label} "{word}" is difficult for a beginner. '
            f'Replace it with a single, simpler, more common {pos_label} '
            f'that keeps the same meaning in the sentence. '
            f'Reply with only the replacement word and nothing else.'
        )
        result = self._generate(prompt, max_new_tokens=8)

        first = result.split()[0].strip(".,;:\"'") if result.split() else ""
        if first.isalpha() and len(first) > 1 and first.lower() != word.lower():
            logger.debug(f"Generated replacement: '{word}' → '{first}'")
            return first

        # If the model didn't produce something useful, leave word unchanged
        logger.debug(f"Generation failed for '{word}' — leaving unchanged.")
        return word

    # ── Public API ────────────────────────────────────────────────────────────

    def simplify_words(self, sentence: str, hard_words: list[dict]) -> dict:
        """
        Takes a sentence and the list of hard-word dicts from
        ComplexityAnalyzer.get_hard_words(), and returns a dict containing:
          simplified   – the sentence with hard words swapped in-place
          replacements – list of {original, replacement, start, end}
          changed      – True if at least one word was actually replaced

        Replacements are applied right-to-left so earlier character offsets
        remain valid even after later characters change.
        """
        words_to_process = hard_words[:_MAX_WORDS_PER_SENTENCE]
        replacements = []

        for hw in words_to_process:
            word      = hw["word"]
            lemma     = hw["lemma"]
            pos       = hw["pos"]
            cache_key = (lemma.lower(), pos)

            if cache_key in self._cache:
                replacement = self._cache[cache_key]
                logger.debug(f"Cache hit: '{word}' → '{replacement}'")
            else:
                # Step 1: ask WordNet for simpler synonyms
                candidates = self._wordnet_candidates(lemma, pos)

                logger.info(
                    f"\n[WORDNET] '{word}' (lemma='{lemma}', pos={pos})\n"
                    f"         Candidates: {candidates if candidates else '(none)'}"
                )

                if candidates:
                    # Step 2a: T5 picks the best candidate in context
                    replacement = self._t5_pick_from_candidates(
                        sentence, word, candidates
                    )
                else:
                    # Step 2b: T5 generates a replacement from scratch
                    replacement = self._t5_generate_replacement(
                        sentence, word, pos
                    )
                self._cache[cache_key] = replacement

            if replacement.lower() != word.lower():
                # Preserve the original word's capitalisation
                if word[0].isupper():
                    replacement = replacement.capitalize()
                replacements.append({
                    "original":    word,
                    "replacement": replacement,
                    "start":       hw["start"],
                    "end":         hw["end"],
                })

        # Rebuild the sentence right-to-left so indices stay valid
        result = sentence
        for r in sorted(replacements, key=lambda x: x["start"], reverse=True):
            result = result[: r["start"]] + r["replacement"] + result[r["end"]:]

        return {
            "simplified":   result,
            "replacements": replacements,
            "changed":      len(replacements) > 0,
        }
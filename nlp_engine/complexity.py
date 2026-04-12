"""
Scores text complexity using CEFR vocabulary levels and word frequency data.

Three tiers for each user level:
  Easy   – words the user already knows comfortably (no action)
  Medium – slightly above the user's level (yellow highlight + definition)
  Hard   – well above the user's level (replaced by the simplifier)

MEDIUM_LEVELS and HARD_LEVELS together define these tiers.  A word at a
MEDIUM level is worth explaining on hover; a word at a HARD level gets
replaced entirely by the simplifier before the user even sees it.
"""

import os
import re
import csv
import glob
import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

import spacy

# ── CEFR ordering ─────────────────────────────────────────────────────────────
CEFR_RANK = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}

# Words AT or ABOVE these CEFR levels are "hard" — the simplifier will try to
# replace them with an easier equivalent.
HARD_LEVELS = {
    "beginner":     {"B2", "C1", "C2"},
    "intermediate": {"C1", "C2"},
    "advanced":     {"C2"},
}

# Words AT these CEFR levels are "medium" — one step above the user's comfort
# zone, worth annotating with a hover definition rather than replacing.
MEDIUM_LEVELS = {
    "beginner":     {"B1"},
    "intermediate": {"B2"},
    "advanced":     {"C1"},
}

# Only content words carry the kind of meaning that benefits from explanation
# or replacement.  Determiners, prepositions, and conjunctions are excluded.
CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


# ── Data-file resolution ───────────────────────────────────────────────────────

def _resolve_data_files() -> tuple[str, str]:
    """
    Return (cefr_csv_path, freq_csv_path).
    Reuses paths written by data_builder.py if available; otherwise
    downloads fresh via kagglehub (result is cached after the first call).
    """
    saved = (
        Path(__file__).parent.parent
        / "transformer_model" / "training_data" / "dataset_paths.json"
    )
    if saved.exists():
        with open(saved) as f:
            paths = json.load(f)
        if Path(paths["cefr_csv"]).exists() and Path(paths["freq_csv"]).exists():
            return paths["cefr_csv"], paths["freq_csv"]

    import kagglehub
    cefr_dir = kagglehub.dataset_download("nezahatkk/10-000-english-words-cerf-labelled")
    freq_dir = kagglehub.dataset_download("rtatman/english-word-frequency")
    cefr_csv = glob.glob(os.path.join(cefr_dir, "**/*.csv"), recursive=True)[0]
    freq_csv = glob.glob(os.path.join(freq_dir,  "**/*.csv"), recursive=True)[0]
    return cefr_csv, freq_csv


def _load_cefr(path: str) -> dict:
    vocab = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            level = row["CEFR"].strip().upper()
            if level in CEFR_RANK:
                vocab[row["headword"].strip().lower()] = level
    print(f"[Complexity] CEFR vocabulary loaded: {len(vocab):,} entries.")
    return vocab


def _load_freq(path: str, top_n: int = 50_000) -> set:
    common: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= top_n:
                break
            common.add(row["word"].strip().lower())
    print(f"[Complexity] Frequency list loaded: top {len(common):,} words.")
    return common


# ── Main class ────────────────────────────────────────────────────────────────

class ComplexityAnalyzer:
    def __init__(self):
        cefr_csv, freq_csv = _resolve_data_files()
        self.cefr_vocab   = _load_cefr(cefr_csv)
        self.common_words = _load_freq(freq_csv)

    @lru_cache(maxsize=16_384)
    def word_level(self, word: str) -> Optional[str]:
        """Return the CEFR level string for a word, or None if not in the list."""
        return self.cefr_vocab.get(word.lower())

    def is_hard_for(self, word: str, user_level: str) -> bool:
        """True if this word is hard enough to warrant replacement."""
        level = self.word_level(word)
        if level:
            return level in HARD_LEVELS.get(user_level, set())
        # Words absent from CEFR: treat as hard only if also absent from
        # the high-frequency list (avoids flagging rare proper names).
        return word.lower() not in self.common_words

    def is_medium_for(self, word: str, user_level: str) -> bool:
        """True if this word is slightly above the user's level (yellow highlight tier)."""
        level = self.word_level(word)
        # Only annotate words explicitly in the CEFR list at the medium level.
        # Words not in the list are either trivially common or proper nouns —
        # neither benefits from a hover definition here.
        return bool(level and level in MEDIUM_LEVELS.get(user_level, set()))

    def get_annotation_words(
        self, text: str, user_level: str, max_words: int = 10
    ) -> list[dict]:
        """
        Returns medium-difficulty words with their exact character positions.

        These are the words that get a yellow underline and an on-hover
        definition card — one step above the user's comfort zone but not
        so hard that the simplifier replaces them outright.

        Each entry in the returned list contains:
          word  – surface form as it appears in the text
          lemma – base form used for WordNet / CEFR lookup
          start – character index in `text` where the word begins
          end   – character index immediately after the word
          pos   – spaCy coarse POS tag (NOUN / VERB / ADJ / ADV)
          level – CEFR level string (e.g. "B1")
        """
        doc = nlp(text)
        annotations = []
        for tok in doc:
            if (
                tok.is_alpha
                and tok.pos_ in CONTENT_POS   # only meaningful content words
                and not tok.is_stop
                and len(tok.text) > 3
                # PROPN is separate from NOUN in spaCy — this check is redundant
                # but makes the intent explicit: no proper nouns.
                and tok.pos_ != "PROPN"
                and self.is_medium_for(tok.lemma_, user_level)
            ):
                annotations.append({
                    "word":  tok.text,
                    "lemma": tok.lemma_,
                    "start": tok.idx,
                    "end":   tok.idx + len(tok.text),
                    "pos":   tok.pos_,
                    "level": self.word_level(tok.lemma_),
                })
            if len(annotations) >= max_words:
                break
        return annotations

    def paragraph_hard_ratio(self, text: str, user_level: str) -> float:
        """Fraction of content words that are 'hard' for this user level."""
        doc = nlp(text)
        content = [
            tok for tok in doc
            if tok.is_alpha and not tok.is_stop and len(tok.text) >= 3
        ]
        if not content:
            return 0.0
        hard = sum(1 for tok in content if self.is_hard_for(tok.lemma_, user_level))
        return hard / len(content)

    def needs_simplification(
        self, text: str, user_level: str, min_hard: int = 1
    ) -> bool:
        """True if text contains at least min_hard words that are hard for this level."""
        doc = nlp(text)
        count = 0
        for tok in doc:
            if tok.is_alpha and self.is_hard_for(tok.lemma_, user_level):
                count += 1
                if count >= min_hard:
                    return True
        return False

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p for p in parts if p]
"""
Loads CEFR vocabulary and word-frequency data, then scores text complexity.

Works at both sentence and paragraph level:
  - needs_simplification(text, level)  → True if text contains hard words
  - paragraph_hard_ratio(text)         → float 0..1 (fraction of hard words)

Data files are located automatically:
  1. If data_builder.py was already run, paths are in
     ../transformer_model/training_data/dataset_paths.json — reused here.
  2. Otherwise downloaded fresh via kagglehub (cached after first run).
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

CEFR_RANK = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}

# For each user level, these CEFR levels are considered "hard"
HARD_LEVELS = {
    "beginner":     {"B2", "C1", "C2"},
    "intermediate": {"C1", "C2"},
    "advanced":     {"C2"},
}

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


# ── Data file resolution ──────────────────────────────────────────────────────

def _resolve_data_files() -> tuple[str, str]:
    """Return (cefr_csv, freq_csv) paths, downloading via kagglehub if needed."""
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
    freq_csv = glob.glob(os.path.join(freq_dir, "**/*.csv"), recursive=True)[0]
    return cefr_csv, freq_csv


def _load_cefr(path: str) -> dict:
    vocab = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            level = row["CEFR"].strip().upper()
            if level in CEFR_RANK:
                vocab[row["headword"].strip().lower()] = level
    return vocab


def _load_freq(path: str, top_n: int = 50_000) -> set:
    common: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= top_n:
                break
            common.add(row["word"].strip().lower())
    return common


# ── Analyzer class ────────────────────────────────────────────────────────────

class ComplexityAnalyzer:
    def __init__(self):
        cefr_csv, freq_csv = _resolve_data_files()
        self.cefr_vocab   = _load_cefr(cefr_csv)
        self.common_words = _load_freq(freq_csv)

    @lru_cache(maxsize=16_384)
    def word_level(self, word: str) -> Optional[str]:
        """Return the CEFR level of a word, or None if unknown."""
        return self.cefr_vocab.get(word.lower())

    def is_hard_for(self, word: str, user_level: str) -> bool:
        """True if this word is above the user's current level."""
        level = self.word_level(word)
        if level:
            return level in HARD_LEVELS.get(user_level, set())
        # Unknown word: treat as hard only if it's also not in the high-frequency list
        return word.lower() not in self.common_words

    def paragraph_hard_ratio(self, text: str, user_level: str) -> float:
        """
        Return the fraction (0..1) of content words in text that are hard
        for the given user level.  A ratio > 0.08 is a good heuristic for
        'this paragraph benefits from simplification'.
        """
        doc = nlp(text)
        content_words = [
            tok for tok in doc
            if tok.is_alpha and not tok.is_stop and len(tok.text) >= 3
        ]
        if not content_words:
            return 0.0
        hard = sum(
            1 for tok in content_words
            if self.is_hard_for(tok.lemma_, user_level)
        )
        return hard / len(content_words)

    def needs_simplification(
        self,
        text: str,
        user_level: str,
        min_hard: int = 1,
    ) -> bool:
        """
        Return True if the text contains at least min_hard words that are
        hard for the given user level.  Used as the gating check before
        calling the (slower) T5 model.
        """
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
        """Regex sentence splitter — kept for backward compatibility."""
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p for p in parts if p]
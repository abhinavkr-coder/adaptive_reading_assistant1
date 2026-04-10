"""
Loads CEFR vocabulary and word-frequency data, then scores sentences.

Data files are located automatically:
  1. If data_builder.py was already run, it saved the paths to
     ../transformer_model/training_data/dataset_paths.json — we reuse those.
  2. Otherwise, we download fresh via kagglehub (cached after first run).
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

CEFR_RANK  = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}
HARD_LEVELS = {
    "beginner":     {"B2", "C1", "C2"},
    "intermediate": {"C1", "C2"},
    "advanced":     {"C2"},
}

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def _resolve_data_files() -> tuple[str, str]:
    """Return (cefr_csv_path, freq_csv_path), downloading if necessary."""
    saved = Path(__file__).parent.parent / "transformer_model" / "training_data" / "dataset_paths.json"
    if saved.exists():
        with open(saved) as f:
            paths = json.load(f)
        if Path(paths["cefr_csv"]).exists() and Path(paths["freq_csv"]).exists():
            return paths["cefr_csv"], paths["freq_csv"]

    # Fresh download via kagglehub (cached locally after the first call)
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
    return vocab


def _load_freq(path: str, top_n: int = 50_000) -> set:
    common: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= top_n:
                break
            common.add(row["word"].strip().lower())
    return common


class ComplexityAnalyzer:
    def __init__(self):
        cefr_csv, freq_csv = _resolve_data_files()
        self.cefr_vocab    = _load_cefr(cefr_csv)
        self.common_words  = _load_freq(freq_csv)

    @lru_cache(maxsize=8192)
    def word_level(self, word: str) -> Optional[str]:
        return self.cefr_vocab.get(word.lower())

    def is_hard_for(self, word: str, user_level: str) -> bool:
        level = self.word_level(word)
        if level:
            return level in HARD_LEVELS.get(user_level, set())
        # Unknown word: hard if it's not in the high-frequency list either
        return word.lower() not in self.common_words

    def needs_simplification(self, sentence: str, user_level: str, min_hard: int = 1) -> bool:
        doc = nlp(sentence)
        count = sum(
            1 for tok in doc
            if tok.is_alpha and self.is_hard_for(tok.lemma_, user_level)
        )
        return count >= min_hard

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p for p in parts if p]
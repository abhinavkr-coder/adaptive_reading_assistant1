"""
Identifies complex words in sentences using CEFR levels and word frequency.

Each hard word is returned with its exact character span (start, end) inside
the sentence so that simplifier.py can do surgical in-place replacement
without ever touching the rest of the text.
"""

import os
import re
import csv
import glob
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import spacy

# ── Logger setup ─────────────────────────────────────────────────────────────
# We use a named logger so its output can be distinguished from Flask's own
# logging in the terminal.
logger = logging.getLogger("ARA.complexity")

CEFR_RANK = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}
HARD_LEVELS = {
    "beginner":     {"B2", "C1", "C2"},
    "intermediate": {"C1", "C2"},
    "advanced":     {"C2"},
}

# Only content words carry meaning that can be simplified.
# Determiners, prepositions, conjunctions etc. must never change.
CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


# ── Data-file resolution ─────────────────────────────────────────────────────

def _resolve_data_files() -> tuple[str, str]:
    """
    Return (cefr_csv_path, freq_csv_path).
    Reuses paths saved by data_builder.py if available; otherwise
    downloads fresh via kagglehub (cached after the first call).
    """
    saved = (
        Path(__file__).parent.parent
        / "transformer_model" / "training_data" / "dataset_paths.json"
    )
    if saved.exists():
        with open(saved) as f:
            paths = json.load(f)
        if Path(paths["cefr_csv"]).exists() and Path(paths["freq_csv"]).exists():
            logger.info("Reusing cached dataset paths from data_builder output.")
            return paths["cefr_csv"], paths["freq_csv"]

    logger.info("Downloading CEFR and frequency datasets via kagglehub…")
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
    logger.info(f"CEFR vocabulary loaded: {len(vocab):,} words.")
    return vocab


def _load_freq(path: str, top_n: int = 50_000) -> set:
    common: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= top_n:
                break
            common.add(row["word"].strip().lower())
    logger.info(f"Frequency list loaded: top {len(common):,} words.")
    return common


# ── Main class ────────────────────────────────────────────────────────────────

class ComplexityAnalyzer:
    def __init__(self):
        cefr_csv, freq_csv = _resolve_data_files()
        self.cefr_vocab   = _load_cefr(cefr_csv)
        self.common_words = _load_freq(freq_csv)

    @lru_cache(maxsize=8192)
    def word_level(self, word: str) -> Optional[str]:
        return self.cefr_vocab.get(word.lower())

    def is_hard_for(self, word: str, user_level: str) -> bool:
        level = self.word_level(word)
        if level:
            return level in HARD_LEVELS.get(user_level, set())
        # Word absent from CEFR list: treat as hard only if also absent
        # from the high-frequency top-50k list (avoids flagging rare names).
        return word.lower() not in self.common_words

    def get_hard_words(self, sentence: str, user_level: str) -> list[dict]:
        """
        Returns one dict per hard content word:
          word  – surface form as it appears in the sentence
          lemma – base form for WordNet / CEFR lookup
          start – character index where the word starts (from spaCy token.idx)
          end   – character index immediately after the word
          pos   – coarse POS tag: NOUN / VERB / ADJ / ADV
          level – CEFR level string, or "unknown" if not in the list

        Stop words and very short tokens are excluded because they carry
        grammatical function rather than vocabulary complexity.
        """
        doc = nlp(sentence)
        hard = []
        for tok in doc:
            if (
                tok.is_alpha
                and tok.pos_ in CONTENT_POS
                and not tok.is_stop
                and len(tok.text) > 3
                and self.is_hard_for(tok.lemma_, user_level)
            ):
                hard.append({
                    "word":  tok.text,
                    "lemma": tok.lemma_,
                    "start": tok.idx,
                    "end":   tok.idx + len(tok.text),
                    "pos":   tok.pos_,
                    "level": self.word_level(tok.lemma_) or "unknown",
                })

        if hard:
            logger.debug(
                f"Hard words in sentence │ level={user_level}\n"
                f"  Sentence : {sentence!r}\n"
                f"  Hard words found: {[w['word'] + '(' + w['level'] + ')' for w in hard]}"
            )
        return hard

    def needs_simplification(self, sentence: str, user_level: str) -> bool:
        return len(self.get_hard_words(sentence, user_level)) > 0

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p for p in parts if p]
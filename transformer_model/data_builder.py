"""
Downloads every dataset automatically and builds (complex, simple) training pairs.

Dataset roles:
  OneStopEnglish  → Gold parallel pairs: Advanced text vs Elementary text of same article.
  Simple Wikipedia → Within-article pairing: teaches the target "very simple" register.
  CCNews          → Within-article pairing on real news: adds topical diversity.
  IMDB reviews    → Within-review pairing: complex film criticism → simple description.
  Movie Subtitles → Within-movie pairing: naturally informal/simple dialogue.

CEFR + Word Frequency CSVs are downloaded here and their paths are saved so the
NLP engine can reuse them without downloading again.
"""

import os
import re
import csv
import json
import glob
import random
from pathlib import Path

import kagglehub
from datasets import load_dataset
from tqdm import tqdm

random.seed(42)

DATA_DIR = Path("./training_data")
DATA_DIR.mkdir(exist_ok=True)

# Paths file lets the NLP engine reuse data already downloaded here.
PATHS_FILE = DATA_DIR / "dataset_paths.json"

HARD_LEVELS = {"B2", "C1", "C2"}


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_cefr(csv_path: str) -> dict:
    vocab = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vocab[row["headword"].strip().lower()] = row["CEFR"].strip().upper()
    return vocab


def split_sentences(text: str) -> list[str]:
    """Lightweight regex sentence splitter — no external NLP needed here."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in parts if len(s.split()) >= 5]


def complexity(sentence: str, cefr: dict) -> float:
    """Fraction of content words at B2 level or above."""
    words = re.findall(r"\b[a-zA-Z]{3,}\b", sentence.lower())
    if not words:
        return 0.0
    return sum(1 for w in words if cefr.get(w, "A1") in HARD_LEVELS) / len(words)


def extract_pair(sentences: list[str], cefr: dict, min_gap: float = 0.08) -> dict | None:
    """
    Within a set of sentences, pair the most complex with the most simple.
    Only create a pair if the complexity gap is meaningful (>= min_gap).
    """
    if len(sentences) < 2:
        return None
    scored = sorted(sentences, key=lambda s: complexity(s, cefr))
    simple, hard = scored[0], scored[-1]
    if hard == simple:
        return None
    if complexity(hard, cefr) - complexity(simple, cefr) >= min_gap:
        return {"source": hard, "target": simple}
    return None


def save_paths(cefr_csv: str, freq_csv: str):
    with open(PATHS_FILE, "w") as f:
        json.dump({"cefr_csv": cefr_csv, "freq_csv": freq_csv}, f)
    print(f"Dataset paths saved → {PATHS_FILE}")


# ── Dataset builders ─────────────────────────────────────────────────────────

def build_onestop(cefr: dict) -> list[dict]:
    """
    OneStopEnglish has 3 levels of the same articles. We sort Advanced (label 2)
    and Elementary (label 0) texts alphabetically so that articles on the same
    topic are roughly aligned, then pair them 1-to-1.
    """
    print("\n[1/5] OneStopEnglish...")
    ds = load_dataset("SetFit/onestop_english", split="train")
    advanced   = sorted(ex["text"] for ex in ds if ex["label"] == 2)
    elementary = sorted(ex["text"] for ex in ds if ex["label"] == 0)
    pairs = [{"source": a, "target": e} for a, e in zip(advanced, elementary)]
    print(f"      {len(pairs)} pairs")
    return pairs


def build_simplewiki(wiki_txt: str, cefr: dict, max_pairs: int = 2000) -> list[dict]:
    """
    Parse AllCombined.txt into articles (title = short line with no period).
    Within each article, pair the most complex sentence with the most simple.
    Even though Simple Wikipedia is already simplified, the within-article variance
    teaches T5 to target the simplest register of English.
    """
    print("\n[2/5] Simple English Wikipedia...")
    pairs: list[dict] = []

    with open(wiki_txt, encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Articles are separated by their title line (≤ 6 words, no period)
    blocks = re.split(r"\n{2,}", content)
    article_sentences: list[str] = []

    for block in tqdm(blocks, desc="  Parsing articles", leave=False):
        block = block.strip()
        if not block:
            continue
        # Title heuristic: short, no sentence-ending punctuation
        if len(block.split()) <= 6 and "." not in block:
            if article_sentences:
                pair = extract_pair(article_sentences, cefr)
                if pair:
                    pairs.append(pair)
                    if len(pairs) >= max_pairs:
                        break
            article_sentences = []
        else:
            article_sentences.extend(split_sentences(block))

    print(f"      {len(pairs)} pairs")
    return pairs


def build_ccnews(cefr: dict, max_pairs: int = 1500) -> list[dict]:
    """
    Stream the 2016 CCNews corpus. For each English article, extract the most
    complex and most simple sentence. We scan up to 8000 articles to fill quota.
    """
    print("\n[3/5] CCNews (streaming)...")
    ds = load_dataset(
        "stanford-oval/ccnews", "2016",
        streaming=True, split="train",
        trust_remote_code=True,
    )
    pairs: list[dict] = []
    scanned = 0

    for article in tqdm(ds, desc="  Scanning articles", total=8000, leave=False):
        if scanned >= 8000 or len(pairs) >= max_pairs:
            break
        scanned += 1
        if article.get("language") != "en" or (article.get("language_score") or 0) < 0.95:
            continue
        text = (article.get("plain_text") or "")[:2500]   # first ~2500 chars is enough
        if len(text) < 100:
            continue
        pair = extract_pair(split_sentences(text), cefr)
        if pair:
            pairs.append(pair)

    print(f"      {len(pairs)} pairs")
    return pairs


def build_imdb(cefr: dict, max_pairs: int = 1500) -> list[dict]:
    """
    IMDB reviews range from terse slang to dense film criticism.
    Within each review, pair the most complex sentence with the simplest.
    """
    print("\n[4/5] IMDB reviews...")
    ds = load_dataset("stanfordnlp/imdb", split="train")
    pairs: list[dict] = []

    for ex in tqdm(ds, desc="  Processing reviews", leave=False):
        # Strip HTML tags that occasionally appear in IMDB reviews
        text = re.sub(r"<[^>]+>", " ", ex["text"])
        pair = extract_pair(split_sentences(text), cefr)
        if pair:
            pairs.append(pair)
        if len(pairs) >= max_pairs:
            break

    print(f"      {len(pairs)} pairs")
    return pairs


def build_subtitles(subtitle_dir: str, cefr: dict, max_pairs: int = 1000) -> list[dict]:
    """
    Movie subtitles are naturally simple/conversational. We group subtitle lines
    by movie (IMDB id column if available) and within each movie do the
    usual complexity-based pairing.
    """
    print("\n[5/5] Movie Subtitles...")
    import pandas as pd

    # Find any CSV or Parquet file in the downloaded directory
    csv_files = glob.glob(os.path.join(subtitle_dir, "**/*.csv"), recursive=True)
    parquet_files = glob.glob(os.path.join(subtitle_dir, "**/*.parquet"), recursive=True)
    files = csv_files or parquet_files

    if not files:
        print("      No subtitle files found — skipping.")
        return []

    pairs: list[dict] = []
    try:
        df = pd.read_parquet(files[0]) if files[0].endswith(".parquet") \
             else pd.read_csv(files[0], on_bad_lines="skip")

        # Identify the text column heuristically
        text_col = next(
            (c for c in df.columns if any(k in c.lower() for k in ("text", "subtitle", "line", "content"))),
            df.columns[0],
        )
        # Identify a grouping column (movie id)
        group_col = next(
            (c for c in df.columns if any(k in c.lower() for k in ("imdb", "movie", "film", "id"))),
            None,
        )

        if group_col:
            for _, group in tqdm(df.groupby(group_col), desc="  By movie", leave=False):
                sents = [str(s) for s in group[text_col].dropna() if len(str(s).split()) >= 3]
                pair = extract_pair(sents, cefr)
                if pair:
                    pairs.append(pair)
                if len(pairs) >= max_pairs:
                    break
        else:
            # No grouping column — use sliding windows of 20 lines as pseudo-documents
            lines = [str(s) for s in df[text_col].dropna() if len(str(s).split()) >= 3]
            for i in range(0, len(lines) - 20, 20):
                pair = extract_pair(lines[i : i + 20], cefr)
                if pair:
                    pairs.append(pair)
                if len(pairs) >= max_pairs:
                    break
    except Exception as exc:
        print(f"      Error reading subtitles ({exc}) — skipping.")
        return []

    print(f"      {len(pairs)} pairs")
    return pairs


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    # -- Download all Kaggle datasets (kagglehub caches locally) --
    print("Downloading Kaggle datasets (cached after first run)...")
    cefr_dir     = kagglehub.dataset_download("nezahatkk/10-000-english-words-cerf-labelled")
    freq_dir     = kagglehub.dataset_download("rtatman/english-word-frequency")
    wiki_dir     = kagglehub.dataset_download("ffatty/plain-text-wikipedia-simpleenglish")
    subtitle_dir = kagglehub.dataset_download("adiamaan/movie-subtitle-dataset")

    # Locate the actual files within the downloaded directories
    cefr_csv = glob.glob(os.path.join(cefr_dir, "**/*.csv"), recursive=True)[0]
    freq_csv = glob.glob(os.path.join(freq_dir,  "**/*.csv"), recursive=True)[0]
    wiki_txt = glob.glob(os.path.join(wiki_dir,  "**/*.txt"), recursive=True)[0]

    # Persist paths so the NLP engine can reuse them without re-downloading
    save_paths(cefr_csv, freq_csv)

    # Load CEFR vocab for complexity scoring
    cefr = load_cefr(cefr_csv)

    # Build pairs from every source
    all_pairs: list[dict] = []
    all_pairs += build_onestop(cefr)
    all_pairs += build_simplewiki(wiki_txt, cefr)
    all_pairs += build_ccnews(cefr)
    all_pairs += build_imdb(cefr)
    all_pairs += build_subtitles(subtitle_dir, cefr)

    # Shuffle and split 90/10
    random.shuffle(all_pairs)
    cut = int(len(all_pairs) * 0.9)

    for split_name, subset in [("train", all_pairs[:cut]), ("val", all_pairs[cut:])]:
        out = DATA_DIR / f"{split_name}.jsonl"
        with open(out, "w") as f:
            for pair in subset:
                f.write(json.dumps(pair) + "\n")
        print(f"Saved {len(subset):,} pairs → {out}")

    print(f"\nTotal pairs: {len(all_pairs):,}  (train {cut:,} / val {len(all_pairs)-cut:,})")


if __name__ == "__main__":
    main()
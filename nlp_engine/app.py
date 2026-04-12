"""
Flask API — bridges the browser extension and the AI models.

Endpoints
---------
GET  /ping          Health check; returns server status and model name.
POST /rewrite       Simplify a batch of paragraphs; also returns annotation
                    data (medium-difficulty words) for yellow highlighting.
GET  /define        Return the CEFR level + WordNet definition for a word.
POST /split         Legacy sentence splitter.
"""

import json
import logging
import logging.config

import nltk
from nltk.corpus import wordnet as wn

from flask import Flask, request, jsonify
from flask_cors import CORS
from complexity import ComplexityAnalyzer
from simplifier import Simplifier

# Download WordNet data once at startup (no-op if already cached)
nltk.download("wordnet",  quiet=True)
nltk.download("omw-1.4", quiet=True)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "ara": {
            "format":  "%(asctime)s  %(levelname)-8s  %(name)-16s  %(message)s",
            "datefmt": "%H:%M:%S",
        }
    },
    "handlers": {
        "console": {
            "class":     "logging.StreamHandler",
            "formatter": "ara",
            "stream":    "ext://sys.stdout",
        }
    },
    "loggers": {
        "ARA":      {"level": "DEBUG", "handlers": ["console"], "propagate": False},
        "werkzeug": {"level": "WARNING"},  # suppress Flask's default request lines
    },
})
logger = logging.getLogger("ARA.app")

# ── App init ───────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

analyzer   = ComplexityAnalyzer()
simplifier = Simplifier()

VALID_LEVELS = {"beginner", "intermediate", "advanced"}

# spaCy → WordNet POS mapping
_WN_POS = {"NOUN": wn.NOUN, "VERB": wn.VERB, "ADJ": wn.ADJ, "ADV": wn.ADV}

# Human-readable POS labels for the hover card
_POS_LABEL = {
    "NOUN": "noun", "VERB": "verb",
    "ADJ":  "adjective", "ADV": "adverb",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _log_json(direction: str, label: str, data: dict):
    pretty   = json.dumps(data, indent=2, ensure_ascii=False)
    indented = "\n".join(f"    {line}" for line in pretty.splitlines())
    logger.info(f"\n{'═'*68}\n  {direction}  {label}\n{'─'*68}\n{indented}\n{'═'*68}")


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/ping")
def ping():
    return jsonify({"status": "ok", "model": simplifier._label})


@app.post("/split")
def split():
    text = request.get_json(force=True).get("text", "")
    return jsonify({"sentences": ComplexityAnalyzer.split_sentences(text)})


@app.get("/define")
def define():
    """
    Return the CEFR level and WordNet definition for a single word.

    Query params:
      word  – the base form (lemma) to look up
      pos   – spaCy POS tag: NOUN | VERB | ADJ | ADV (optional)

    Response:
      { word, level, pos, definition, example }
    """
    word = request.args.get("word", "").strip().lower()
    pos  = request.args.get("pos",  "").strip().upper()

    if not word:
        return jsonify({"error": "word parameter required"}), 400

    logger.info(f"[/define] word='{word}' pos='{pos}'")

    level   = analyzer.word_level(word) or "unknown"
    wn_pos  = _WN_POS.get(pos)

    # Get synsets; restrict to the given POS if provided, else search all
    synsets = wn.synsets(word, pos=wn_pos) if wn_pos else wn.synsets(word)

    if not synsets:
        response = {
            "word":       word,
            "level":      level,
            "pos":        _POS_LABEL.get(pos, pos.lower() or "unknown"),
            "definition": "No definition found in WordNet.",
            "example":    None,
        }
        logger.info(f"[/define] No synsets found for '{word}'")
        return jsonify(response)

    # Use the first (most common) synset
    synset     = synsets[0]
    definition = synset.definition()
    examples   = synset.examples()

    response = {
        "word":       word,
        "level":      level,
        "pos":        _POS_LABEL.get(pos, pos.lower() or "unknown"),
        "definition": definition,
        "example":    examples[0] if examples else None,
    }

    logger.info(
        f"[/define] '{word}' → level={level}, "
        f"def='{definition[:60]}…'"
    )
    return jsonify(response)


@app.post("/rewrite")
def rewrite():
    """
    Rewrite paragraphs and annotate medium-difficulty words.

    Request body:
        { "paragraphs": [...], "user_level": "beginner" }

    Response body — each result contains:
        original     – unchanged input paragraph
        rewritten    – simplified version (or original if no change needed)
        changed      – True if the text was actually rewritten
        annotations  – list of medium-difficulty words with character positions
                       in the `rewritten` text, for yellow in-page highlighting
    """
    body       = request.get_json(force=True)
    paragraphs = body.get("paragraphs", [])
    user_level = body.get("user_level", "beginner").strip().lower()

    _log_json("→ IN", "/rewrite REQUEST", {
        "user_level":      user_level,
        "paragraph_count": len(paragraphs),
        "preview":         (paragraphs[0][:80] + "…") if paragraphs else "",
    })

    if not paragraphs:
        return jsonify({"error": "No paragraphs provided."}), 400
    if user_level not in VALID_LEVELS:
        return jsonify({"error": f"user_level must be one of {sorted(VALID_LEVELS)}."}), 400

    results = []
    for raw in paragraphs:
        text = (raw or "").strip()

        if len(text) < 30:
            results.append({
                "original":    raw,
                "rewritten":   raw,
                "changed":     False,
                "annotations": [],
            })
            continue

        # ── Complexity gate ────────────────────────────────────────────────
        # Only send to the GPU when the paragraph has hard vocabulary.
        needs_rewrite = analyzer.needs_simplification(text, user_level, min_hard=2)

        if needs_rewrite:
            try:
                rewritten = simplifier.simplify(text, user_level)
            except Exception as exc:
                logger.error(f"Simplifier error: {exc}")
                rewritten = text
        else:
            rewritten = text

        changed = rewritten.strip() != text.strip()

        # Annotate medium-difficulty words in whichever text will be displayed
        text_for_annotation = rewritten if changed else text
        annotations = analyzer.get_annotation_words(text_for_annotation, user_level)

        logger.info(
            f"[/rewrite] changed={changed} | "
            f"annotations={len(annotations)} | "
            f"text='{text[:50]}…'"
        )

        results.append({
            "original":    raw,
            "rewritten":   rewritten,
            "changed":     changed,
            "annotations": annotations,
        })

    _log_json("← OUT", "/rewrite RESPONSE", {
        "results_count":   len(results),
        "changed_count":   sum(1 for r in results if r["changed"]),
        "annotation_total": sum(len(r["annotations"]) for r in results),
    })

    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
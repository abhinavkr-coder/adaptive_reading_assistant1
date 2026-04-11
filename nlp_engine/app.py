"""
Flask API — bridges the browser extension and the AI models.

All JSON flowing in and out of each endpoint is logged to the terminal
so you can watch the full pipeline in action.
"""

import json
import logging
import logging.config

from flask import Flask, request, jsonify
from flask_cors import CORS

from complexity import ComplexityAnalyzer
from simplifier import Simplifier

# ── Logging configuration ─────────────────────────────────────────────────────
# A single configuration block keeps all loggers consistent.
# We suppress Flask/Werkzeug's own verbose request lines and replace them
# with our structured JSON logs instead.
logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "ara": {
            # Format: timestamp  LEVEL  logger-name  message
            "format": "%(asctime)s  %(levelname)-8s  %(name)-20s  %(message)s",
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
        "ARA":       {"level": "DEBUG", "handlers": ["console"], "propagate": False},
        "werkzeug":  {"level": "WARNING"},   # suppress Flask's default request lines
    },
})

logger = logging.getLogger("ARA.app")

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# Build the analyzer first so the Simplifier can reuse its loaded vocab maps,
# avoiding the cost of parsing the same CSVs twice.
analyzer   = ComplexityAnalyzer()
simplifier = Simplifier(
    cefr_vocab=analyzer.cefr_vocab,
    common_words=analyzer.common_words,
)


# ── Helper: pretty-print a dict to the terminal ───────────────────────────────
def _log_json(direction: str, label: str, data: dict):
    """
    Prints a clearly bordered JSON block to the terminal.
    direction is either '→ IN' (incoming) or '← OUT' (outgoing).
    """
    pretty = json.dumps(data, indent=2, ensure_ascii=False)
    # Indent every line of the JSON so it sits neatly under the header
    indented = "\n".join(f"    {line}" for line in pretty.splitlines())
    logger.info(f"\n{'═'*70}\n  {direction}  {label}\n{'─'*70}\n{indented}\n{'═'*70}")


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/ping")
def ping():
    return jsonify({"status": "ok"})


@app.post("/split")
def split():
    body = request.get_json(force=True)
    _log_json("→ IN", "/split  REQUEST", body)

    text      = body.get("text", "")
    sentences = ComplexityAnalyzer.split_sentences(text)
    response  = {"sentences": sentences}

    _log_json("← OUT", "/split  RESPONSE", response)
    return jsonify(response)


@app.post("/simplify")
def simplify():
    body = request.get_json(force=True)
    _log_json("→ IN", "/simplify  REQUEST", body)

    sentences  = body.get("sentences", [])
    user_level = body.get("user_level", "beginner").lower()

    if not sentences:
        return jsonify({"error": "No sentences provided."}), 400
    if user_level not in ("beginner", "intermediate", "advanced"):
        return jsonify({"error": "Invalid user_level."}), 400

    results = []
    for sent in sentences:
        # Step 1: NLP engine identifies hard words with character positions
        hard_words = analyzer.get_hard_words(sent, user_level)

        if hard_words:
            logger.info(
                f"\n[NLP → SIMPLIFIER]\n"
                f"  Sentence   : {sent!r}\n"
                f"  Hard words : {json.dumps(hard_words, ensure_ascii=False)}"
            )
            # Step 2: Simplifier replaces only those words using WordNet + flan-T5
            outcome = simplifier.simplify_words(sent, hard_words)
            results.append({
                "original":     sent,
                "simplified":   outcome["simplified"],
                "changed":      outcome["changed"],
                "replacements": outcome["replacements"],
            })
            logger.info(
                f"\n[SIMPLIFIER → RESULT]\n"
                f"  Original   : {sent!r}\n"
                f"  Simplified : {outcome['simplified']!r}\n"
                f"  Swaps      : {outcome['replacements']}"
            )
        else:
            results.append({
                "original":     sent,
                "simplified":   sent,
                "changed":      False,
                "replacements": [],
            })

    response = {"results": results}
    _log_json("← OUT", "/simplify  RESPONSE", response)
    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
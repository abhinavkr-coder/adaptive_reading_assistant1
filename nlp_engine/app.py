"""
Flask API — bridges the browser extension and the AI models.

Endpoints
---------
GET  /ping          Health check; also warms up the T5 model on first call.
POST /rewrite       Main endpoint. Accepts a batch of paragraphs + user_level,
                    returns a rewritten version of each paragraph.
POST /split         Legacy sentence splitter (kept for compatibility).

Architecture note
-----------------
/rewrite operates at paragraph level (not sentence level).  This lets the
model maintain pronoun references, topic continuity, and natural flow across
sentences — something sentence-by-sentence simplification cannot do.

The complexity gate in needs_simplification() prevents sending simple
paragraphs (e.g. a short byline or nav label) to the GPU, so inference
only runs where it actually matters.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from complexity import ComplexityAnalyzer
from simplifier import Simplifier

app = Flask(__name__)
CORS(app)

# Both objects are heavy; instantiate once at startup.
analyzer   = ComplexityAnalyzer()
simplifier = Simplifier()

VALID_LEVELS = {"beginner", "intermediate", "advanced"}


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/ping")
def ping():
    """Health check. Also confirms the model is loading correctly."""
    return jsonify({"status": "ok", "model": simplifier._label})


@app.post("/split")
def split():
    """Legacy: split raw text into sentences (used by old content.js)."""
    text = request.get_json(force=True).get("text", "")
    return jsonify({"sentences": ComplexityAnalyzer.split_sentences(text)})


@app.post("/rewrite")
def rewrite():
    """
    Rewrite a batch of paragraphs at the requested reading level.

    Request body (JSON):
        {
            "paragraphs": ["paragraph one text ...", "paragraph two text ...", ...],
            "user_level": "beginner" | "intermediate" | "advanced"
        }

    Response body (JSON):
        {
            "results": [
                {
                    "original":  "...",
                    "rewritten": "...",
                    "changed":   true | false
                },
                ...
            ]
        }

    The "changed" flag is false when either:
      (a) the complexity gate decides no simplification is needed, or
      (b) the model's output is identical to the input.
    This lets the frontend skip highlighting on already-simple paragraphs.
    """
    body       = request.get_json(force=True)
    paragraphs = body.get("paragraphs", [])
    user_level = body.get("user_level", "beginner").strip().lower()

    if not paragraphs:
        return jsonify({"error": "No paragraphs provided."}), 400
    if user_level not in VALID_LEVELS:
        return jsonify({"error": f"user_level must be one of {sorted(VALID_LEVELS)}."}), 400

    results = []
    for raw in paragraphs:
        text = (raw or "").strip()

        # Skip empty strings or very short fragments (captions, labels, etc.)
        if len(text) < 30:
            results.append({"original": raw, "rewritten": raw, "changed": False})
            continue

        # Complexity gate: only call the GPU when the paragraph actually has
        # vocabulary that is hard for this user level.
        if not analyzer.needs_simplification(text, user_level, min_hard=1):
            results.append({"original": raw, "rewritten": raw, "changed": False})
            continue

        try:
            rewritten = simplifier.simplify(text, user_level)
        except Exception as exc:
            # Log but don't crash — return original so the page stays readable.
            app.logger.error(f"Simplifier error: {exc}")
            results.append({"original": raw, "rewritten": raw, "changed": False})
            continue

        changed = rewritten.strip() != text.strip()
        results.append({
            "original":  raw,
            "rewritten": rewritten,
            "changed":   changed,
        })

    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
"""Flask API bridging the browser extension and the AI models."""

from flask import Flask, request, jsonify
from flask_cors import CORS
from complexity import ComplexityAnalyzer
from simplifier import Simplifier

app = Flask(__name__)
CORS(app)

analyzer   = ComplexityAnalyzer()
simplifier = Simplifier()


@app.get("/ping")
def ping():
    return jsonify({"status": "ok"})


@app.post("/split")
def split():
    text = request.get_json(force=True).get("text", "")
    return jsonify({"sentences": ComplexityAnalyzer.split_sentences(text)})


@app.post("/simplify")
def simplify():
    body       = request.get_json(force=True)
    sentences  = body.get("sentences", [])
    user_level = body.get("user_level", "beginner").lower()

    if not sentences:
        return jsonify({"error": "No sentences provided."}), 400
    if user_level not in ("beginner", "intermediate", "advanced"):
        return jsonify({"error": "Invalid user_level."}), 400

    results = []
    for sent in sentences:
        if analyzer.needs_simplification(sent, user_level):
            simplified = simplifier.simplify(sent)
            results.append({"original": sent, "simplified": simplified,
                            "changed": simplified.strip() != sent.strip()})
        else:
            results.append({"original": sent, "simplified": sent, "changed": False})

    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
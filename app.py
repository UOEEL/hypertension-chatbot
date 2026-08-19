from flask import Flask, render_template, request, jsonify

from rag import (
    retrieve,
    has_enough_information,
    check_answerability,
    generate_answer,
)

import config


app = Flask(__name__)


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")


# =========================================================
# ASK
# =========================================================

@app.route("/ask", methods=["POST"])
def ask():

    data = request.get_json()

    question = data.get("question", "").strip()

    if not question:

        return jsonify({
            "answer": "Please enter a question.",
            "sources": []
        })


    # =====================================================
    # RETRIEVE
    # =====================================================

    try:

        hits = retrieve(
            question,
            top_k=config.DEFAULT_TOP_K
        )

    except Exception as e:

        print("Retrieval error:", e)

        return jsonify({
            "answer": "Sorry, an error occurred while searching the documents.",
            "sources": []
        })


    # =====================================================
    # FIRST REFUSAL GATE
    # =====================================================

    if not has_enough_information(hits):

        return jsonify({
            "answer": (
                "I don't have enough information in the "
                "provided documents to accurately answer "
                "your question."
            ),
            "sources": []
        })


    # =====================================================
    # SECOND REFUSAL GATE
    # =====================================================

    try:

        answerable = check_answerability(
            question,
            hits
        )

    except Exception as e:

        print("Answerability error:", e)

        answerable = False


    if not answerable:

        return jsonify({
            "answer": (
                "I don't have enough information in the "
                "provided documents to accurately answer "
                "your question."
            ),
            "sources": []
        })


    # =====================================================
    # GENERATE ANSWER
    # =====================================================

    try:

        answer = generate_answer(
            question,
            hits
        )

    except Exception as e:

        print("Generation error:", e)

        return jsonify({
            "answer": (
                "Sorry, an error occurred while generating "
                "the answer."
            ),
            "sources": []
        })


    # =====================================================
    # SOURCES
    # =====================================================

    sources = []

    for hit in hits:

        metadata = hit.get("metadata", {})

        sources.append({
            "page": metadata.get("page_number"),
            "section": metadata.get("section_title"),
            "score": round(
                hit.get("score", 0),
                3
            )
        })


    # =====================================================
    # RETURN RESPONSE
    # =====================================================

    return jsonify({
        "answer": answer,
        "sources": sources
    })


# =========================================================
# RUN SERVER
# =========================================================

if __name__ == "__main__":
    import os

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
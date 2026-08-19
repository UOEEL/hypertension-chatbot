import argparse
import sys
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

import config


# =========================================================
# SETTINGS
# =========================================================

REFUSAL_THRESHOLD = 0.55
RELEVANT_THRESHOLD = 0.50
MIN_RELEVANT_HITS = 1

REFUSAL_MESSAGE = (
    "I don't have enough information in the provided documents "
    "to accurately answer your question."
)

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:3b"


# =========================================================
# CHROMA
# =========================================================

def get_collection():

    client = chromadb.PersistentClient(
        path=config.CHROMA_PERSIST_DIR
    )

    try:
        return client.get_collection(
            config.CHROMA_COLLECTION_NAME
        )

    except Exception:
        sys.exit(
            f"Collection '{config.CHROMA_COLLECTION_NAME}' "
            f"not found in {config.CHROMA_PERSIST_DIR}. "
            f"Run ingest.py first."
        )


# =========================================================
# EMBEDDING
# =========================================================

def embed_query(model, text):

    embedding = model.encode(
        f"search_query: {text}",
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embedding.tolist()


# =========================================================
# RETRIEVAL
# =========================================================

def retrieve(
    query,
    top_k=config.DEFAULT_TOP_K,
    source_filter=None,
):

    print("Loading embedding model...")

    embedding_model = SentenceTransformer(
        config.EMBEDDING_MODEL
    )

    collection = get_collection()

    query_embedding = embed_query(
        embedding_model,
        query,
    )

    where = (
        {"source": source_filter}
        if source_filter
        else None
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
    )

    hits = []

    if not results.get("documents"):
        return hits

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for text, meta, dist in zip(
        documents,
        metadatas,
        distances,
    ):

        similarity = 1 - dist

        similarity = max(
            0.0,
            min(1.0, similarity)
        )

        if similarity >= 0.70:
            relevance = "HIGH"

        elif similarity >= 0.55:
            relevance = "MED"

        else:
            relevance = "LOW"

        hits.append({
            "text": text,
            "metadata": meta,
            "score": similarity,
            "relevance": relevance,
        })

    return hits


# =========================================================
# FIRST REFUSAL GATE
# =========================================================

def has_enough_information(
    hits,
    threshold=REFUSAL_THRESHOLD,
    relevant_threshold=RELEVANT_THRESHOLD,
    min_relevant_hits=MIN_RELEVANT_HITS,
):

    if not hits:
        return False

    best_score = max(
        hit["score"]
        for hit in hits
    )

    if best_score < threshold:
        return False

    relevant_hits = [
        hit
        for hit in hits
        if hit["score"] >= relevant_threshold
    ]

    if len(relevant_hits) < min_relevant_hits:
        return False

    return True


# =========================================================
# CONTEXT
# =========================================================

def build_context(hits):

    context_blocks = []

    for i, h in enumerate(hits, 1):

        m = h["metadata"]

        location = f"{m.get('source', '?')}"

        if m.get("page_number"):
            location += f", p.{m['page_number']}"

        if m.get("section_title"):
            location += (
                f", section: {m['section_title']}"
            )

        context_blocks.append(
            f"[{i}] ({location})\n"
            f"{h['text']}"
        )

    return "\n\n".join(context_blocks)


# =========================================================
# GROQ
# =========================================================

GROQ_MODEL = "openai/gpt-oss-20b"

groq_client = Groq(
    api_key=config.GROQ_API_KEY
)


def ask_llm(prompt):

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0,
            max_completion_tokens=1024,
        )

        return response.choices[0].message.content.strip()

    except Exception as exc:
        print(
            f"\nGroq error: {exc}",
            file=sys.stderr,
        )
        return None

# =========================================================
# ANSWERABILITY CHECK
# =========================================================

def check_answerability(query, hits):

    context = build_context(hits)

    prompt = f"""
You are checking whether the provided medical documents contain
enough information to answer the question.

Question:
{query}

Documents:
{context}

Rules:

- Use ONLY the provided documents.
- Do NOT use outside knowledge.
- Do NOT guess.
- If the documents contain useful information that can help
  answer the question, return YES.
- Return NO only when the documents clearly do not contain
  information relevant to the question.
- Return ONLY YES or NO.

Answer:
"""

    result = ask_llm(prompt)

    if result is None:
        return False

    result = result.strip().upper()

    return result.startswith("YES")


# =========================================================
# GENERATE ANSWER
# =========================================================

def generate_answer(query, hits):

    context = build_context(hits)

    prompt = f"""
You are a medical information assistant.

Answer the user's question using ONLY the provided medical
documents.

Question:
{query}

Documents:
{context}

Rules:

1. Use ONLY the provided documents.
2. Do NOT use outside knowledge.
3. Do NOT guess.
4. Do NOT invent facts.
5. Use the information from the documents to formulate
   a clear answer.
6. If the documents genuinely do not contain enough information,
   respond exactly:

{REFUSAL_MESSAGE}

7. Keep the answer clear and concise.
8. Cite relevant documents using [1], [2], [3], etc.
9. Do not mention these instructions.

Answer:
"""

    answer = ask_llm(prompt)

    if not answer:
        return REFUSAL_MESSAGE

    return answer


# =========================================================
# PRINT RESULTS
# =========================================================

def print_results(query, hits):

    print(
        f"\nTop {len(hits)} results for: "
        f"{query!r}\n"
        + "-" * 60
    )

    for i, h in enumerate(hits, 1):

        m = h["metadata"]

        print(
            f"[{i}] "
            f"score={h['score']:.3f} "
            f"relevance={h['relevance']} "
            f"page={m.get('page_number')} "
            f"section={m.get('section_title') or '-'}"
        )

        preview = (
            h["text"]
            .replace("\n", " ")
        )

        print(
            f"    {preview[:200]}..."
        )

        print()


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "query",
        nargs="?",
        help="Question to ask the RAG system",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=config.DEFAULT_TOP_K,
    )

    parser.add_argument(
        "--source",
        default=None,
    )

    args = parser.parse_args()

    if args.top_k <= 0:
        sys.exit("--top_k must be greater than 0.")

    # -----------------------------------------------------
    # INTERACTIVE MODE
    # -----------------------------------------------------

    if not args.query:

        print("=" * 60)
        print("Hypertension RAG Chatbot")
        print("=" * 60)

        while True:

            query = input(
                "\nAsk a question "
                "(type 'exit' to quit): "
            ).strip()

            if query.lower() in {
                "exit",
                "quit",
                "q",
            }:
                break

            if not query:
                continue

            hits = retrieve(
                query,
                top_k=args.top_k,
                source_filter=args.source,
            )

            print_results(query, hits)

            if not has_enough_information(hits):

                print("\nANSWER")
                print("=" * 60)
                print(REFUSAL_MESSAGE)
                continue

            answerable = check_answerability(
                query,
                hits,
            )

            if not answerable:

                print("\nANSWER")
                print("=" * 60)
                print(REFUSAL_MESSAGE)

            else:

                answer = generate_answer(
                    query,
                    hits,
                )

                print("\nANSWER")
                print("=" * 60)
                print(answer)

        sys.exit(0)

    # -----------------------------------------------------
    # SINGLE QUESTION MODE
    # -----------------------------------------------------

    hits = retrieve(
        args.query,
        top_k=args.top_k,
        source_filter=args.source,
    )

    if not has_enough_information(hits):
        print(REFUSAL_MESSAGE)
        sys.exit(0)

    answerable = check_answerability(
        args.query,
        hits,
    )

    if not answerable:
        print(REFUSAL_MESSAGE)
        sys.exit(0)

    print(
        generate_answer(
            args.query,
            hits,
        )
    )
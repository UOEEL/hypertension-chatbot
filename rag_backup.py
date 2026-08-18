import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer

import config


# =========================================================
# REFUSAL SETTINGS
# =========================================================

REFUSAL_THRESHOLD = config.REFUSAL_THRESHOLD
RELEVANT_THRESHOLD = config.RELEVANT_THRESHOLD
MIN_RELEVANT_HITS = config.MIN_RELEVANT_HITS

REFUSAL_MESSAGE = config.REFUSAL_MESSAGE


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    config.EMBEDDING_MODEL
)


# =========================================================
# CHROMA COLLECTION
# =========================================================

def get_collection():

    client = chromadb.PersistentClient(
        path=config.CHROMA_PERSIST_DIR
    )

    return client.get_collection(
        config.CHROMA_COLLECTION_NAME
    )


# =========================================================
# EMBED QUERY
# =========================================================

def embed_query(query: str):

    embedding = embedding_model.encode(
        f"search_query: {query}",
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embedding.tolist()


# =========================================================
# RETRIEVE
# =========================================================

def retrieve(
    query: str,
    top_k: int = config.DEFAULT_TOP_K,
):

    collection = get_collection()

    query_embedding = embed_query(
        query
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    hits = []

    if not results.get("documents"):
        return hits

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for text, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):

        similarity = 1 - distance

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

        hits.append(
            {
                "text": text,
                "metadata": metadata,
                "score": similarity,
                "relevance": relevance,
            }
        )

    return hits


# =========================================================
# FIRST REFUSAL GATE
# =========================================================

def has_enough_information(
    hits,
):

    if not hits:
        return False

    best_score = max(
        hit["score"]
        for hit in hits
    )

    # Best retrieved result is too weak
    if best_score < REFUSAL_THRESHOLD:
        return False

    relevant_hits = [
        hit
        for hit in hits
        if hit["score"] >= RELEVANT_THRESHOLD
    ]

    if len(relevant_hits) < MIN_RELEVANT_HITS:
        return False

    return True


# =========================================================
# BUILD CONTEXT
# =========================================================

def build_context(hits):

    context_blocks = []

    for i, hit in enumerate(
        hits,
        start=1,
    ):

        metadata = hit["metadata"]

        source = metadata.get(
            "source",
            "?",
        )

        page = metadata.get(
            "page_number"
        )

        section = metadata.get(
            "section_title"
        )

        location = source

        if page:
            location += f", p.{page}"

        if section:
            location += (
                f", section: {section}"
            )

        context_blocks.append(
            f"[{i}] ({location})\n"
            f"{hit['text']}"
        )

    return "\n\n".join(
        context_blocks
    )


# =========================================================
# ANSWERABILITY CHECK
# =========================================================

def check_answerability(
    client,
    query,
    hits,
):

    context = build_context(
        hits
    )

    prompt = f"""
You are evaluating whether a question can be answered accurately
using ONLY the provided documents.

Question:
{query}

Documents:
{context}

Rules:

1. Use ONLY the provided documents.
2. Do not use outside knowledge.
3. Do not guess.
4. Do not make assumptions.
5. The documents must contain enough information to directly answer
   the question.
6. If the documents are only generally related but do not actually
   answer the question, return NO.
7. If important information is missing, return NO.

Return ONLY:

YES

or

NO
"""

    try:

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0,
        )

        result = (
            response
            .choices[0]
            .message
            .content
            .strip()
            .upper()
        )

        return result == "YES"

    except Exception as error:

        print(
            f"Answerability check failed: {error}"
        )

        # Fail closed
        return False


# =========================================================
# GENERATE ANSWER
# =========================================================

def generate_answer(
    client,
    query,
    hits,
):

    context = build_context(
        hits
    )

    prompt = f"""
You are a medical information assistant.

Answer the user's question using ONLY the provided context.

Question:
{query}

Context:
{context}

Important rules:

1. Use ONLY the provided context.
2. Do not use outside knowledge.
3. Do not guess.
4. Do not invent facts.
5. Every factual claim must be supported by the context.
6. If the context does not contain enough information, respond exactly:

"{REFUSAL_MESSAGE}"

7. Cite the relevant context using [1], [2], [3], etc.
8. Keep the answer clear and concise.
9. Do not mention information that is not supported by the context.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0,
    )

    return (
        response
        .choices[0]
        .message
        .content
        .strip()
    )


# =========================================================
# MAIN RAG FUNCTION
# =========================================================

def ask(
    query: str,
    top_k: int = config.DEFAULT_TOP_K,
):

    # -----------------------------------------------------
    # Retrieve
    # -----------------------------------------------------

    hits = retrieve(
        query,
        top_k=top_k,
    )

    # -----------------------------------------------------
    # FIRST GATE
    # Retrieval relevance
    # -----------------------------------------------------

    if not has_enough_information(
        hits
    ):

        return {
            "answer": REFUSAL_MESSAGE,
            "hits": hits,
            "refused": True,
        }

    # -----------------------------------------------------
    # OpenAI client
    # -----------------------------------------------------

    client = OpenAI(
        api_key=config.OPENAI_API_KEY
    )

    # -----------------------------------------------------
    # SECOND GATE
    # Answerability
    # -----------------------------------------------------

    answerable = check_answerability(
        client,
        query,
        hits,
    )

    if not answerable:

        return {
            "answer": REFUSAL_MESSAGE,
            "hits": hits,
            "refused": True,
        }

    # -----------------------------------------------------
    # Generate final answer
    # -----------------------------------------------------

    answer = generate_answer(
        client,
        query,
        hits,
    )

    return {
        "answer": answer,
        "hits": hits,
        "refused": False,
    }


# =========================================================
# COMMAND LINE TEST
# =========================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("Hypertension RAG Chatbot")
    print("=" * 60)

    question = input(
        "\nAsk a question: "
    ).strip()

    if not question:

        print(
            "Please enter a question."
        )

        raise SystemExit

    result = ask(
        question
    )

    print()
    print("=" * 60)
    print("ANSWER")
    print("=" * 60)

    print(
        result["answer"]
    )

    print()
    print("=" * 60)
    print("RETRIEVED SOURCES")
    print("=" * 60)

    for i, hit in enumerate(
        result["hits"],
        start=1,
    ):

        metadata = hit["metadata"]

        print(
            f"[{i}] "
            f"score={hit['score']:.3f} "
            f"relevance={hit['relevance']} "
            f"page={metadata.get('page_number')} "
            f"section={metadata.get('section_title')}"
        )
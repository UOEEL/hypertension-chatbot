import argparse
import hashlib
import re
import sys
from pathlib import Path

import chromadb
import pymupdf
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import config


SUPPORTED_EXTS = {".pdf"}


# =========================================================
# FIND PDF FILES
# =========================================================

def find_documents(path: Path):

    if path.is_file():

        if path.suffix.lower() != ".pdf":
            sys.exit(f"Not a PDF: {path}")

        return [path]

    return sorted(
        p
        for p in path.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTS
    )


# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(text: str) -> str:

    if not text:
        return ""

    text = text.replace("\xa0", " ")

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(r"[ \t]+", " ", text)

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# =========================================================
# EXTRACT PDF
# =========================================================

def extract_pdf(filepath: Path):

    pages = []

    doc = pymupdf.open(filepath)

    for page_number, page in enumerate(doc, start=1):

        text = clean_text(
            page.get_text("text")
        )

        raw_blocks = page.get_text("blocks")

        blocks = []

        for block in raw_blocks:

            if len(block) < 5:
                continue

            x0, y0, x1, y1, block_text = block[:5]

            block_text = clean_text(
                block_text
            )

            if not block_text:
                continue

            blocks.append(
                {
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "text": block_text,
                }
            )

        pages.append(
            {
                "page_number": page_number,
                "text": text,
                "blocks": blocks,
            }
        )

    doc.close()

    return pages


# =========================================================
# DETECT HEADINGS
# =========================================================

def looks_like_heading(text: str) -> bool:

    text = text.strip()

    if not text:
        return False

    if len(text) > 180:
        return False

    if len(text) < 3:
        return False

    if re.fullmatch(r"[\d\s.\-]+", text):
        return False

    if " | " in text and "p." in text.lower():
        return False

    if text.startswith("http://") or text.startswith("https://"):
        return False

    sentence_count = len(
        re.findall(
            r"[.!?](?:\s|$)",
            text
        )
    )

    if sentence_count >= 3:
        return False

    words = text.split()

    if len(words) > 25:
        return False

    # Numbered headings
    if re.match(
        r"^\d+(?:\.\d+)*[\s.)\-]+[A-Za-z]",
        text
    ):
        return True

    # Chapter / section / part
    if re.match(
        r"^(chapter|section|part|annex|appendix)\b",
        text,
        re.IGNORECASE,
    ):
        return True

    # ALL CAPS
    letters = [
        c
        for c in text
        if c.isalpha()
    ]

    if letters:

        uppercase_ratio = (
            sum(c.isupper() for c in letters)
            / len(letters)
        )

        if (
            uppercase_ratio > 0.75
            and len(words) <= 15
        ):
            return True

    # Short title-like text
    if len(words) <= 10:

        if not text.endswith(
            (".", "?", "!")
        ):
            return True

    return False


# =========================================================
# EXTRACT SECTIONS
# =========================================================

def extract_sections(pages):

    segments = []

    current_section = ""

    current_text = []

    current_page = None

    for page in pages:

        page_number = page["page_number"]

        for block in page["blocks"]:

            text = block["text"].strip()

            if not text:
                continue

            if looks_like_heading(text):

                if current_text:

                    combined = clean_text(
                        "\n".join(current_text)
                    )

                    if combined:

                        segments.append(
                            {
                                "text": combined,
                                "page_number": current_page,
                                "section_title": current_section,
                            }
                        )

                current_section = text

                current_text = []

                current_page = page_number

            else:

                if current_page is None:
                    current_page = page_number

                current_text.append(text)

    # Save final section
    if current_text:

        combined = clean_text(
            "\n".join(current_text)
        )

        if combined:

            segments.append(
                {
                    "text": combined,
                    "page_number": current_page,
                    "section_title": current_section,
                }
            )

    return segments


# =========================================================
# SPLIT LARGE TEXT
# =========================================================

def split_large_text(
    text: str,
    max_characters: int,
    overlap: int,
):

    text = clean_text(text)

    if len(text) <= max_characters:
        return [text]

    paragraphs = [
        p.strip()
        for p in re.split(
            r"\n{2,}",
            text
        )
        if p.strip()
    ]

    chunks = []

    current = ""

    for paragraph in paragraphs:

        if len(paragraph) > max_characters:

            sentences = re.split(
                r"(?<=[.!?])\s+",
                paragraph,
            )

            for sentence in sentences:

                sentence = sentence.strip()

                if not sentence:
                    continue

                if (
                    len(current)
                    + len(sentence)
                    + 1
                    <= max_characters
                ):

                    if current:
                        current += " " + sentence
                    else:
                        current = sentence

                else:

                    if current:
                        chunks.append(
                            current.strip()
                        )

                    if len(sentence) > max_characters:

                        start = 0

                        while start < len(sentence):

                            end = (
                                start
                                + max_characters
                            )

                            piece = sentence[
                                start:end
                            ].strip()

                            if piece:
                                chunks.append(
                                    piece
                                )

                            start = end - overlap

                            if start < 0:
                                start = 0

                        current = ""

                    else:

                        current = sentence

        else:

            if (
                len(current)
                + len(paragraph)
                + 2
                <= max_characters
            ):

                if current:

                    current += (
                        "\n\n"
                        + paragraph
                    )

                else:

                    current = paragraph

            else:

                if current:

                    chunks.append(
                        current.strip()
                    )

                current = paragraph

    if current:
        chunks.append(
            current.strip()
        )

    # Add overlap
    if (
        overlap <= 0
        or len(chunks) <= 1
    ):
        return chunks

    overlapped = [chunks[0]]

    for i in range(1, len(chunks)):

        previous = chunks[i - 1]

        overlap_text = previous[
            -overlap:
        ]

        combined = (
            overlap_text
            + "\n\n"
            + chunks[i]
        )

        overlapped.append(
            combined.strip()
        )

    return overlapped


# =========================================================
# CREATE CHUNKS
# =========================================================

def section_aware_chunks(
    pages,
    source_name: str,
):

    sections = extract_sections(pages)

    records = []

    for section in sections:

        text = section["text"].strip()

        if not text:
            continue

        section_title = (
            section["section_title"].strip()
        )

        chunks = split_large_text(
            text,
            max_characters=(
                config.CHUNK_MAX_CHARACTERS
            ),
            overlap=config.CHUNK_OVERLAP,
        )

        for chunk_text in chunks:

            if not chunk_text.strip():
                continue

            record_id = hashlib.sha256(
                (
                    f"{source_name}-"
                    f"{section['page_number']}-"
                    f"{len(records)}-"
                    f"{chunk_text[:100]}"
                ).encode("utf-8")
            ).hexdigest()

            records.append(
                {
                    "id": record_id,
                    "text": chunk_text,
                    "metadata": {
                        "source": source_name,
                        "page_number": (
                            section["page_number"]
                        ),
                        "section_title": (
                            section_title
                        ),
                        "element_category": "Text",
                    },
                }
            )

    return records


# =========================================================
# EMBEDDINGS
# =========================================================

def embed_texts(
    model: SentenceTransformer,
    texts: list[str],
):

    embeddings = []

    for i in tqdm(
        range(
            0,
            len(texts),
            config.EMBED_BATCH_SIZE,
        ),
        desc="  embedding",
    ):

        batch = texts[
            i:i + config.EMBED_BATCH_SIZE
        ]

        batch_embeddings = model.encode(
            [
                f"search_document: {text}"
                for text in batch
            ],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        embeddings.extend(
            batch_embeddings.tolist()
        )

    return embeddings


# =========================================================
# CHROMA
# =========================================================

def get_collection(reset: bool = False):

    client = chromadb.PersistentClient(
        path=config.CHROMA_PERSIST_DIR
    )

    if reset:

        try:

            client.delete_collection(
                config.CHROMA_COLLECTION_NAME
            )

        except Exception:
            pass

    return client.get_or_create_collection(
        name=config.CHROMA_COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine"
        },
    )


# =========================================================
# INGEST
# =========================================================

def ingest(
    path_str: str,
    reset: bool = False,
):

    print(
        f"Loading embedding model: "
        f"{config.EMBEDDING_MODEL}"
    )

    embedding_model = (
        SentenceTransformer(
            config.EMBEDDING_MODEL
        )
    )

    path = Path(path_str)

    docs = find_documents(path)

    if not docs:

        sys.exit(
            f"No supported documents found "
            f"at {path_str}"
        )

    collection = get_collection(
        reset=reset
    )

    for doc_path in docs:

        print(
            f"\nProcessing: "
            f"{doc_path.name}"
        )

        try:

            pages = extract_pdf(
                doc_path
            )

        except Exception as e:

            print(
                f"  FAILED to parse "
                f"{doc_path.name}: {e}"
            )

            continue

        print(
            f"  extracted "
            f"{len(pages)} pages"
        )

        records = section_aware_chunks(
            pages,
            source_name=doc_path.name,
        )

        if not records:

            print(
                "  no chunks extracted, "
                "skipping"
            )

            continue

        print(
            f"  created "
            f"{len(records)} chunks"
        )

        texts = [
            r["text"]
            for r in records
        ]

        embeddings = embed_texts(
            embedding_model,
            texts,
        )

        collection.upsert(
            ids=[
                r["id"]
                for r in records
            ],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                r["metadata"]
                for r in records
            ],
        )

        print(
            f"  upserted "
            f"{len(records)} chunks"
        )

    print(
        f"\nDone."
    )

    print(
        f"Collection: "
        f"{config.CHROMA_COLLECTION_NAME}"
    )

    print(
        f"Total chunks: "
        f"{collection.count()}"
    )

    print(
        f"Database: "
        f"{config.CHROMA_PERSIST_DIR}"
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "path",
        help="PDF file or folder containing PDFs",
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete old Chroma collection first",
    )

    args = parser.parse_args()

    ingest(
        args.path,
        reset=args.reset,
    )
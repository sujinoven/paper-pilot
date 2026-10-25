from pathlib import Path

import chromadb

from paperpilot.ingest import parse_pdf
from paperpilot.chunking import chunk_paper
from paperpilot.vector_store import PaperVectorStore


PDF_PATH = Path(
    "data/papers/attention_is_all_you_need.pdf"
)

CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "paperpilot_papers"


def main():
    # Parse the paper using the updated section detector.
    paper = parse_pdf(PDF_PATH)

    chunks = chunk_paper(
        paper,
        chunk_size=220,
        overlap=40,
    )

    print("Paper:", paper.title)
    print("Chunks generated:", len(chunks))

    # WARNING: This deletes the entire named collection,
    # including any other papers indexed in it.
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("Deleted old collection.")
    except chromadb.errors.NotFoundError:
        print("Collection does not exist yet.")

    # PaperVectorStore recreates the collection.
    store = PaperVectorStore()

    store.add_chunks(chunks)

    print("Chunks stored:", store.count())

    assert store.count() == len(chunks)

    # Confirm that the corrected section metadata
    # is present in the persisted collection.
    result = store.collection.get(
        ids=[f"{paper.paper_id}-17"],
        include=["metadatas"],
    )

    if result["ids"]:
        print("Chunk 17 metadata:", result["metadatas"][0])

    print("Rebuild complete.")


if __name__ == "__main__":
    main()

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from .chunking import Chunk


MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "paperpilot_papers"


class PaperVectorStore:
    def __init__(
        self,
        persist_directory: str = "data/chroma",
    ):
        # Load the embedding model.
        self.model = SentenceTransformer(
            MODEL_NAME,
            device="cpu",
        )

        # Create or reopen a persistent database.
        db_path = Path(persist_directory)
        db_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(db_path),
        )

        # Use cosine distance for similarity search.
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            configuration={"hnsw": {"space": "cosine"}},
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Generate embeddings and store paper chunks."""

        if not chunks:
            return

        texts = [chunk.text for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
        )

        ids = [chunk.chunk_id for chunk in chunks]

        metadatas = [
            {
                "paper_id": chunk.paper_id,
                "title": chunk.title,
                "section": chunk.section,
                "page": chunk.page,
            }
            for chunk in chunks
        ]

        # Upsert inserts new chunks and updates existing IDs.
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )

    def search(
        self,
        question: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Find chunks semantically related to a question."""

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        if self.collection.count() == 0:
            return []

        # BGE recommends this prefix for retrieval queries.
        query_text = (
            "Represent this sentence for searching "
            "relevant passages: "
            + question
        )

        query_embedding = self.model.encode(
            [query_text],
            normalize_embeddings=True,
        )[0]

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self.collection.count()),
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        matches = []

        for chunk_id, text, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            matches.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "metadata": metadata,
                    "distance": distance,
                }
            )

        return matches

    def count(self) -> int:
        """Return the number of stored chunks."""
        return self.collection.count()

from pathlib import Path
import re

import streamlit as st

from paperpilot.ingest import parse_pdf
from paperpilot.chunking import chunk_paper
from paperpilot.vector_store import PaperVectorStore
from paperpilot.retrieval import retrieve
from paperpilot.generation import PaperAnswerGenerator


st.set_page_config(
    page_title="PaperPilot",
    page_icon="📚",
    layout="wide",
)

st.title("📚 PaperPilot")
st.caption("Upload research papers and ask questions using local AI.")


PAPERS_DIR = Path("data/papers")
PAPERS_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_resource
def load_resources():
    store = PaperVectorStore()
    generator = PaperAnswerGenerator(model="qwen2.5:3b")
    return store, generator


def safe_pdf_filename(filename: str) -> str:
    """Create a safe filename for locally saved PDFs."""
    name = Path(filename.replace("\\", "/")).name
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", Path(name).stem)
    return f"{stem or 'uploaded_paper'}.pdf"


try:
    store, generator = load_resources()
except Exception as error:
    st.error(f"Failed to initialize PaperPilot: {error}")
    st.stop()


if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_paper" not in st.session_state:
    st.session_state.active_paper = None


with st.sidebar:
    st.header("Research Papers")

    uploaded_file = st.file_uploader(
        "Upload a research paper",
        type=["pdf"],
    )

    if uploaded_file is not None:
        if st.button("Index Paper", use_container_width=True):
            try:
                with st.spinner("Parsing and indexing PDF..."):
                    filename = safe_pdf_filename(uploaded_file.name)
                    pdf_path = PAPERS_DIR / filename

                    pdf_path.write_bytes(uploaded_file.getvalue())

                    paper = parse_pdf(pdf_path)
                    chunks = chunk_paper(paper)

                    if not chunks:
                        raise ValueError(
                            "No searchable text was extracted from this PDF."
                        )

                    store.add_chunks(chunks)

                    st.session_state.active_paper = {
                        "paper_id": paper.paper_id,
                        "title": paper.title,
                        "filename": filename,
                        "chunk_count": len(chunks),
                    }

                    st.session_state.messages = []

                st.success(
                    f"Indexed '{paper.title}' "
                    f"({len(chunks)} chunks)."
                )

            except Exception as error:
                st.error(f"Failed to index paper: {error}")

    st.divider()

    st.header("Current Paper")

    if st.session_state.active_paper:
        active_paper = st.session_state.active_paper

        st.success(active_paper["title"])
        st.caption(
            f"Chunks indexed: {active_paper['chunk_count']}"
        )
    else:
        st.info(
            "Upload and index a PDF to select a paper. "
            "Your previously indexed papers remain searchable."
        )

    st.write(f"Total chunks in database: {store.count()}")

    st.divider()

    st.header("Settings")

    top_k = st.slider(
        "Number of source passages",
        min_value=1,
        max_value=5,
        value=3,
    )

    st.caption("Model: qwen2.5:3b (Ollama)")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            for source in message.get("sources", []):
                with st.expander(
                    f"Source [{source['number']}] — "
                    f"{source['section']} "
                    f"(Page {source['page']})"
                ):
                    st.write(source["text"])


question = st.chat_input("Ask a question about the paper...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching and generating answer..."):
                results = retrieve(
                    store,
                    question,
                    top_k=top_k,
                    candidate_k=15,
                )

                answer = generator.generate(
                    question=question,
                    results=results,
                )

            st.markdown(answer)

            sources = []

            for index, result in enumerate(results, start=1):
                metadata = result["metadata"]

                source = {
                    "number": index,
                    "title": metadata.get("title", "Unknown Paper"),
                    "section": metadata.get(
                        "section", "Unknown Section"
                    ),
                    "page": metadata.get("page", "?"),
                    "text": result["text"],
                }

                sources.append(source)

                with st.expander(
                    f"Source [{index}] — "
                    f"{source['title']} — "
                    f"{source['section']} "
                    f"(Page {source['page']})"
                ):
                    st.write(source["text"])

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                }
            )

        except Exception as error:
            st.error(f"An error occurred: {error}")
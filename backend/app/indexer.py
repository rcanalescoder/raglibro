from __future__ import annotations

from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_ollama import OllamaEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # rag-libro/backend/app cl-> rag-libro
PDF_PATH = PROJECT_ROOT / "libro.pdf"

# Directorio donde se persistirá el índice
INDEX_DIR = Path(__file__).resolve().parents[1] / "storage" / "faiss_index"


def build_or_load_vectorstore(
    *,
    embedding_model: str = "mxbai-embed-large",
    force_rebuild: bool = False,
) -> FAISS:
    """
    Construye (o carga) un vector store FAISS.

    - Si el índice existe en disco y no se fuerza rebuild, se carga.
    - Si no existe, se lee el PDF, se parte en chunks, se embebe y se guarda.

    embedding_model:
      - Modelo de embeddings servido por Ollama (local).
      - Ejemplo: "mxbai-embed-large" (modelo de embeddings popular en Ollama).
    """
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {PDF_PATH}. Copiad el libro como 'libro.pdf' en la raíz del proyecto."
        )

    embeddings = OllamaEmbeddings(model=embedding_model)

    if INDEX_DIR.exists() and not force_rebuild:
        # allow_dangerous_deserialization=True:
        #   - FAISS usa pickle para persistencia de metadata.
        #   - En un entorno real conviene controlar muy bien de dónde vienen estos archivos.
        #   - Para un ejemplo local, es aceptable.
        return FAISS.load_local(
            folder_path=str(INDEX_DIR),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )

    # 1) Cargar PDF como lista de Document (texto + metadatos, típicamente página)
    loader = PyPDFLoader(str(PDF_PATH))
    docs = loader.load()

    # 2) Trocear en chunks: esencial para RAG, porque un libro entero no cabe en contexto
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(docs)

    # 3) Construir FAISS desde documentos
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 4) Guardar en disco para no recalcular embeddings en cada arranque
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))

    return vectorstore

if __name__ == "__main__":
    """
    Ejecución rápida para validar que:
    - se encuentra libro.pdf
    - se crea (o carga) el índice FAISS
    - se guarda en backend/storage/faiss_index
    """
    from time import perf_counter

    t0 = perf_counter()
    vs = build_or_load_vectorstore(force_rebuild=False)
    t1 = perf_counter()

    # 'index' es el índice FAISS interno
    print("✅ Vectorstore FAISS listo.")
    print(f"   - Nº vectores (chunks): {vs.index.ntotal}")
    print(f"   - Índice en disco: {INDEX_DIR}")
    print(f"   - Tiempo: {t1 - t0:.2f}s")
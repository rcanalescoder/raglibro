from indexer import build_or_load_vectorstore

if __name__ == "__main__":
    vs = build_or_load_vectorstore()

    query = "¿Cómo se crean equipos de alto rendimiento?"
    docs = vs.similarity_search(query, k=5)

    print(f"Consulta: {query}\n")
    for i, d in enumerate(docs, 1):
        page = d.metadata.get("page", "?")
        snippet = d.page_content[:300].replace("\n", " ")
        print(f"[{i}] pág. {page} -> {snippet}...")
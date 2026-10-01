from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

from app.config import settings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={
        "device": "cpu",
    },
    encode_kwargs={
        "normalize_embeddings": True,
    },
)



qdrant_client = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key,
)




vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=settings.qdrant_collection,
    embedding=embeddings,
)




retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 50,
    }
)




reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)



def search_document(query: str) -> list[dict]:
    """
    Search the Qdrant document collection.

    Pipeline:

        Query
          ↓
        Vector retrieval
          ↓
        50 candidates
          ↓
        BGE reranking
          ↓
        Duplicate-page removal
          ↓
        Top 4 unique results
    """

    
    documents = retriever.invoke(query)

    if not documents:
        return []

    

    pairs = [
        (
            query,
            document.page_content,
        )
        for document in documents
    ]

    

    scores = reranker.predict(pairs)

    
    ranked_documents = sorted(
        zip(documents, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    
    unique_documents = []

    seen_pages = set()

    for document, score in ranked_documents:

        # Get page number from metadata.
        page = document.metadata.get(
            "page_number",
            document.metadata.get(
                "page",
                "unknown",
            ),
        )

        # If this page has already been selected,
        # skip this document.
        if page in seen_pages:
            continue

        # Remember this page.
        seen_pages.add(page)

        # Keep document + reranking score.
        unique_documents.append(
            (
                document,
                score,
            )
        )

        # Stop after 4 unique pages.
        if len(unique_documents) >= 4:
            break

    
    results = []

    for document, score in unique_documents:

        source = document.metadata.get(
            "source_file",
            document.metadata.get(
                "source",
                "unknown",
            ),
        )

        page = document.metadata.get(
            "page_number",
            document.metadata.get(
                "page",
                "unknown",
            ),
        )

        content_type = document.metadata.get(
            "content_type",
            "unknown",
        )

        results.append(
            {
                "text": document.page_content,
                "source": source,
                "page": page,
                "content_type": content_type,
                "rerank_score": float(score),
            }
        )

    return results
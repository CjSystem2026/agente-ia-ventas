"""
Motor RAG (Retrieval-Augmented Generation) para el Agente de Ventas IA.
Utiliza ChromaDB como base de datos vectorial y los embeddings de Gemini
para indexar y recuperar documentos de conocimiento relevantes.
"""

import os
import logging
import uuid
import chromadb
from chromadb.config import Settings
import google.generativeai as genai

logger = logging.getLogger(__name__)

# Directorio persistente para ChromaDB (mapeado al volumen Docker ./data)
CHROMA_DATA_DIR = "/app/data/chroma_db"
COLLECTION_NAME = "bingo_knowledge"

# Cliente global de ChromaDB (singleton)
_chroma_client = None
_collection = None


def _get_collection():
    """Obtiene o inicializa la colección de ChromaDB (singleton thread-safe)."""
    global _chroma_client, _collection
    if _collection is not None:
        return _collection

    os.makedirs(CHROMA_DATA_DIR, exist_ok=True)
    _chroma_client = chromadb.PersistentClient(
        path=CHROMA_DATA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    _collection = _chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    logger.info(f"ChromaDB inicializado. Colección '{COLLECTION_NAME}' tiene {_collection.count()} documentos.")
    return _collection


def _embed_text(text: str) -> list[float]:
    """Genera el vector de embeddings para un texto usando Gemini."""
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_document"
    )
    return result["embedding"]


def _embed_query(text: str) -> list[float]:
    """Genera el vector de embeddings para una consulta (query) usando Gemini."""
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_query"
    )
    return result["embedding"]


def add_document(title: str, content: str) -> dict:
    """
    Indexa un documento en ChromaDB dividiéndolo en chunks si es largo.
    Retorna el ID del documento creado.
    """
    collection = _get_collection()

    # Dividir el texto en chunks de ~500 caracteres con solapamiento de 50
    chunks = _split_into_chunks(content, chunk_size=500, overlap=50)
    doc_id = str(uuid.uuid4())

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}_chunk_{i}"
        try:
            embedding = _embed_text(chunk)
        except Exception as e:
            logger.error(f"Error al generar embedding para chunk {i} del doc '{title}': {e}")
            continue

        ids.append(chunk_id)
        embeddings.append(embedding)
        documents.append(chunk)
        metadatas.append({
            "doc_id": doc_id,
            "title": title,
            "chunk_index": i,
            "total_chunks": len(chunks)
        })

    if ids:
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        logger.info(f"Documento '{title}' indexado con {len(ids)} chunks. doc_id={doc_id}")

    return {"doc_id": doc_id, "title": title, "chunks": len(ids)}


def search_knowledge(query: str, n_results: int = 3) -> str:
    """
    Busca los fragmentos más relevantes en la base de conocimiento
    para una consulta dada. Retorna el contexto como texto plano.
    """
    collection = _get_collection()

    if collection.count() == 0:
        return ""

    try:
        query_embedding = _embed_query(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        if not results or not results["documents"] or not results["documents"][0]:
            return ""

        context_parts = []
        seen_titles = set()

        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            # Filtrar resultados con distancia coseno > 0.7 (baja relevancia)
            if dist > 0.7:
                continue

            title = meta.get("title", "Documento")
            if title not in seen_titles:
                seen_titles.add(title)
                context_parts.append(f"[{title}]\n{doc}")

        return "\n\n".join(context_parts)

    except Exception as e:
        logger.error(f"Error al buscar en ChromaDB: {e}")
        return ""


def list_documents() -> list[dict]:
    """Lista todos los documentos únicos indexados en la base de conocimiento."""
    collection = _get_collection()

    if collection.count() == 0:
        return []

    try:
        # Obtener todos los metadatos
        results = collection.get(include=["metadatas"])
        docs_map = {}

        for meta in results["metadatas"]:
            doc_id = meta.get("doc_id")
            if doc_id and doc_id not in docs_map:
                docs_map[doc_id] = {
                    "doc_id": doc_id,
                    "title": meta.get("title", "Sin título"),
                    "total_chunks": meta.get("total_chunks", 1)
                }

        return list(docs_map.values())
    except Exception as e:
        logger.error(f"Error al listar documentos: {e}")
        return []


def delete_document(doc_id: str) -> bool:
    """Elimina todos los chunks de un documento por su doc_id."""
    collection = _get_collection()
    try:
        collection.delete(where={"doc_id": doc_id})
        logger.info(f"Documento doc_id={doc_id} eliminado de ChromaDB.")
        return True
    except Exception as e:
        logger.error(f"Error al eliminar documento {doc_id}: {e}")
        return False


def _split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Divide un texto largo en chunks con solapamiento para mejor recuperación."""
    if len(text) <= chunk_size:
        return [text.strip()]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Intentar cortar en el último punto o salto de línea para no partir palabras
        if end < len(text):
            last_break = max(chunk.rfind(". "), chunk.rfind("\n"), chunk.rfind(". "))
            if last_break > chunk_size // 2:
                chunk = chunk[:last_break + 1]
                end = start + last_break + 1

        chunks.append(chunk.strip())
        start = end - overlap

    return [c for c in chunks if c]

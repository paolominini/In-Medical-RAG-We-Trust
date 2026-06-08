"""
Creation and storage of the embedded vectors
This manages both the semantic similarity search and the
(eventual) callback to the actual documents to be retrived 
(if the search fails retrieving them)  
"""

from __future__ import annotations

import chromadb

from src.config import RAGConfig
from src.embedder import EmbeddingModel
from src.schema import Document, RetrievedContext


class ChromaStore:
    """
    Manages the local ChromaDB instnace.
    Every document is also stored with `entity_id` and `label` in its metadata: 
    it will allow to inject the missing documents when the retrivial misses the
    actual documents needed.
    """

    def __init__(self, config: RAGConfig, embedder: EmbeddingModel):
        self.config = config
        self.embedder = embedder

        # A persistent client saves the db under config.chroma_dir,
        # so we don't run it for every experiment
        self._client = chromadb.PersistentClient(path=str(config.chroma_dir))
        self._collection = self._get_or_create_collection()

    def _get_or_create_collection(self): 
        """Initializes the collection using cosine similarity."""
        return self._client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    
    # Indexing
    
    def add(self, documents: list[Document]) -> None:
        """
        Computes dense embeddings via the injected EmbeddingModel and ingests 
        documents into ChromaDB. Appends entity_id and label as metadata for 
        subsequent targeted retrieval.
        """
        if not documents:
            return

        texts = [doc.text for doc in documents]
        embeddings = self.embedder.encode(texts)

        self._collection.add(
            ids=[doc.doc_id for doc in documents],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {"entity_id": doc.entity_id, "label": doc.label}
                for doc in documents
            ],
        )

    def reset(self) -> None:
        """
        Drops and recreates the collection, 
        only for when we recreate the dataset.
        """
        self._client.delete_collection(name=self.config.collection_name)
        self._collection = self._get_or_create_collection()

    
    # Retrieval
    # ----------------------
    # Real Semantic Retrieval 
    def query(self, text: str, top_k: int | None = None) -> RetrievedContext:
        """
        Performs semantic similarity search.
        Embeds the query and retrieves the top-k nearest documents.
        """
        k = top_k if top_k is not None else self.config.top_k
        query_embedding = self.embedder.encode([text])

        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=k,
        )
        return RetrievedContext(query=text, documents=self._unpack_query_results(results))
    # ----------------------
    # Deterministic (FallBack) Retrieval
    def get_by_entity(self, entity_id: str) -> list[Document]:
        """
        Performs deterministic retrieval using metadata filtering.
        Essential for the "forced injection" mechanism to ensure both 
        conflicting documents are present in the final prompt context.
        """
        results = self._collection.get(
            where={"entity_id": entity_id},
            include=["documents", "metadatas"],
        )
        return self._unpack_get_results(results)
    # we will integrate in other modules the way in which these two retrievial interact
        
    # ------------------------------------------------------------------
    # Helpers
    # Chroma returns plain dicts; these functions are meant to turn them
    # into our set format Document. 
    
    @staticmethod
    def _unpack_query_results(results: dict) -> list[Document]:
        """Transforms nested ChromaDB query results into typed 'Document' objects."""
        ids = results["ids"][0]
        texts = results["documents"][0]
        metadatas = results["metadatas"][0]
        return [
            Document(doc_id=doc_id, entity_id=meta["entity_id"], label=meta["label"], text=text)
            for doc_id, text, meta in zip(ids, texts, metadatas)
        ]

    @staticmethod
    def _unpack_get_results(results: dict) -> list[Document]:
        """Transforms flat ChromaDB get results into typed Document objects."""
        ids = results["ids"]
        texts = results["documents"]
        metadatas = results["metadatas"]
        return [
            Document(doc_id=doc_id, entity_id=meta["entity_id"], label=meta["label"], text=text)
            for doc_id, text, meta in zip(ids, texts, metadatas)
        ]
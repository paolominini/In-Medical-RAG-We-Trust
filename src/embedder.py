"""
Embedding model wrapper.
Abstracts the sentence-transformers library to provide a clean interface
for generating dense vector representations of text.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import RAGConfig


class EmbeddingModel:
    """
    Wraps the dense embedding model.
    Strategically allocated to the CPU to preserve 
    unified memory (MPS) for the heavier generative LLM.
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        # Explicitly pin to CPU to save VRAM for the generation model
        self._model = SentenceTransformer(config.embed_id, device="cpu")

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Embeds strings into dense vectors, 
        returning a NumPy array of shape (len(texts), embedding_dim),
        aligning with ChromaDB's expected format.
        """
        return self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    @property
    def dimension(self) -> int:
        """Size of the vectors this model produces (384 for MiniLM-L6-v2)."""
        return self._model.get_embedding_dimension()

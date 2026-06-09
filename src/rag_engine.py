"""
Orchestration layer for the RAG experimental pipeline.

RAGEngine.answer() executes one complete RAG cycle:
    retrieve → ensure conflict pair → build prompt → generate → parse

The injected prompt is the sole changeable variable: 
everything else is fixed (store, llm and 'config.py' stated values).
"""

from __future__ import annotations

import logging

from src.config import RAGConfig
from src.llm import LLMClient
from src.prompts import PromptStrategy
from src.schema import Document, QueryResult, RetrievedContext
from src.vector_store import ChromaStore

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    Connects together ChromaStore, LLMClient, and a PromptStrategy.

    Constructor takes all dependencies explicitly so that
    ExperimentRunner can build two engines that are identical 
    except for the prompting strategy.
    """

    def __init__(
        self,
        store: ChromaStore,
        llm: LLMClient,
        strategy: PromptStrategy,
        config: RAGConfig,
    ) -> None:
        self._store = store
        self._llm = llm
        self._strategy = strategy
        self._config = config

    # Public API

    def answer(self, entity_id: str, question: str) -> QueryResult:
        """
        Execute the full RAG pipeline for a single (entity_id, question) pair.

        Pipeline stages:
          1. Semantic retrieval: embed the question, fetch top_k nearest
             documents from the 200-document corpus.
          2. Conflict-pair guarantee: ensure both the healthy and poisoned
             document for the target entity are present in the context window.
          3. Prompt construction: delegate to the injected PromptStrategy.
          4. Generation: invoke the LLM under greedy decoding.
          5. Parsing: extract the structured output envelope or fail gracefully.
        """
        # Stage 1 — dense retrieval over the full corpus (200 docs)
        initial_context: RetrievedContext = self._store.query(question)

        # Stage 2 — conflict-pair integrity check
        final_docs, retrieval_complete, injection_used = self._ensure_conflict_pair(
            entity_id, initial_context.documents
        )
            #_ensure_conflict_pair() is defined later in this module
        final_context = RetrievedContext(query=question, documents=final_docs)

        logger.debug(
            "entity=%s  strategy=%s  retrieval_complete=%s  injection_used=%s",
            entity_id,
            self._strategy.name,
            retrieval_complete,
            injection_used,
        )

        # Stage 3 — prompt construction (strategy-specific)
        messages = self._strategy.build_messages(question, final_docs)

        # Stage 4 — greedy generation (temperature=0, do_sample=False)
        raw_output: str = self._llm.generate(messages)

        # Stage 4 — greedy generation (temperature=0, do_sample=False)
        generation = self._strategy.parse_output(raw_output)

        return QueryResult(
            entity_id=entity_id,
            question=question,
            strategy_name=self._strategy.name,
            context=final_context,
            retrieval_complete=retrieval_complete,
            injection_used=injection_used,
            generation=generation,
        )

    def run_batch(self, queries: list[tuple[str, str]]) -> list[QueryResult]:
        """
        Execute the pipeline over a list of (entity_id, question) pairs.

        Args:
            queries: sequence of (entity_id, question) pairs, typically
                     produced by CorpusBuilder.get_queries().

        Returns:
            One QueryResult per input pair, preserving insertion order.
        """
        results: list[QueryResult] = []
        total = len(queries)

        for i, (entity_id, question) in enumerate(queries, start=1):
            print(f"[{i}/{total}] [{self._strategy.name}] {entity_id}: {question[:70]}")
            results.append(self.answer(entity_id, question))

        return results

    # ------------------------------------------------------------------
    # Private helpers

    def _ensure_conflict_pair(
        self,
        entity_id: str,
        retrieved_docs: list[Document],
    ) -> tuple[list[Document], bool, bool]:
        """
        After semantic retrieval, checks whether both the healthy AND the poisoned
        document for this entity are present in the returned list.

        If either is missing, it fetches the entity's documents directly via
        ChromaDB's metadata filter (no semantic ranking — exact lookup by entity_id)
        and appends the missing one(s) to the context.

        This guarantees the model always sees both sides of the conflict,
        so any difference in behaviour is attributable to the prompt strategy
        rather than to one side of the conflict being invisible.

        Also records two side-metrics:
          retrieval_complete: True if semantic search naturally surfaced both docs
          injection_used:     True if we had to force-inject the missing one

        Returns:
            (final_docs, retrieval_complete, injection_used)
        """
        # Which labels for THIS entity did semantic search return?
        found_labels: set[str] = {
            doc.label
            for doc in retrieved_docs
            if doc.entity_id == entity_id
        }

        both_present = {"healthy", "poisoned"}.issubset(found_labels)

        if both_present:
            # Semantic search did the right thing 
            return retrieved_docs, True , False #True for retrieval complete, False for injection_used

        # Fetch all docs for this entity by exact metadata filter
        entity_docs: list[Document] = self._store.get_by_entity(entity_id)

        # Only append whichever label(s) are actually missing
        missing_docs: list[Document] = [
            doc for doc in entity_docs
            if doc.label not in found_labels
        ]

        return retrieved_docs + missing_docs, False, True #False for retrieval complete, True for injection_used
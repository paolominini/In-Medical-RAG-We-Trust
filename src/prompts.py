"""
Prompt Strategy Module.

Implements the Strategy Pattern to dynamically inject task framing 
(Baseline vs. Epistemic Verification) into the RAG pipeline while 
maintaining a symmetric output parser for accurate metric tracking.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod

from src.schema import Document, GenerationOutput


class PromptStrategy(ABC):
    """
    Abstract base class defining the interface for prompt injection.

    Subclasses isolate the instructional manipulation (the prompt) to ensure
    all other pipeline variables (retrieval, model, decoding parameters) remain frozen.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the experimental condition (e.g.'baseline')."""
        ...

    @abstractmethod
    def build_messages(self, question: str, docs: list[Document]) -> list[dict[str, str]]:
        """
        Constructs the strict conversational turns required by the LLM.

        Args:
            question: The user's query.
            docs: The retrieved contextual documents (guaranteed to contain conflicting pairs).

        Returns:
            A list of dictionary objects conforming to the chat template schema.
        """
        ...

    def parse_output(self, raw: str) -> GenerationOutput:
        """
        One json.loads() attempt on the model's raw response.

        Success path: extract answer, conflict_detected, and (optionally) reasoning.
        Failure path: set parse_failed=True and keep raw for downstream lexicon matching.

        No regex repair, No fallback JSON extraction. The error will be taken into account.
        """
        try:
            data: dict = json.loads(raw)
            return GenerationOutput(
                answer=str(data.get("answer", "")),
                conflict_detected=bool(data.get("conflict_detected", False)),
                reasoning=data.get("reasoning", None),  # only Epistemic populates this
                raw=raw,
                parse_failed=False,
            )
        except (json.JSONDecodeError, AttributeError):
            # Record the instruction-following failure (the llm didn't produce a proper JSON) 
            # without discarding the raw output to allow for fallback lexical evaluation.
            return GenerationOutput(
                answer="",
                conflict_detected=False,
                reasoning=None,
                raw=raw,
                parse_failed=True,
            )

    @staticmethod
    def _format_documents(docs: list[Document]) -> str:
        """Serializes Document objects into a structured textual block for contextual injection."""
        return "\n\n".join(
            f"[Document {i}]\n{doc.text}"
            for i, doc in enumerate(docs, start=1)
        )


class BaselineStrategy(PromptStrategy):
    """
    Baseline arm: simply asks the model to answer the question from context.
    No instruction to check for consistency or flag conflicts.

    This is the control condition, it represents a typical RAG prompt.
    """

    @property
    def name(self) -> str:
        return "baseline"

    def build_messages(self, question: str, docs: list[Document]) -> list[dict[str, str]]:
        system = (
            "You are a medical information assistant. "
            "Answer the question using ONLY the information in the provided documents. "
            "Output a single JSON object with no markdown and no code blocks. "
            'Use this exact format: {"answer": "<short answer>", "conflict_detected": true or false}'
        )
        user = (
            f"Documents:\n{self._format_documents(docs)}"
            f"\n\nQuestion: {question}"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]


class EpistemicVerificationStrategy(PromptStrategy):
    """
    Epistemic arm: instructs the model to actively verify source consistency
    and explicitly flag contradictions before committing to an answer.

    The meta-prompt that should push the model from passive generation 
    towards critical verification.
    It includes a 'reasoning' field in the JSON so the model can show its work.
    """

    @property
    def name(self) -> str:
        return "epistemic_verification"

    def build_messages(self, question: str, docs: list[Document]) -> list[dict[str, str]]:
        system = (
            "You are a critical medical information analyst. "
            "Before answering, carefully compare the provided documents to each other. "
            "If any documents contain contradictory or inconsistent claims about the same fact, "
            "you MUST set conflict_detected to true and explain the contradiction in 'reasoning'. "
            "If you cannot determine a reliable answer because sources conflict, "
            "set answer to 'cannot determine'. "
            "Output a single JSON object with no markdown and no code blocks. "
            'Use this exact format: {"answer": "<short answer or cannot determine>", '
            '"conflict_detected": true or false, '
            '"reasoning": "<brief explanation of what the documents say and whether they agree>"}'
        )
        user = (
            f"Documents:\n{self._format_documents(docs)}"
            f"\n\nQuestion: {question}"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
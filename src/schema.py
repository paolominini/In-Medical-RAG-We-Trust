"""
Data contracts for the pipeline.
Defines the strictly typed structures passed between modules to ensure
consistency across data ingestion, retrieval, generation, and evaluation.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MedicalEntity:
    """
    A single entry from the fictional dataset.
    Represents an invented medical entity, the contested attribute, 
    and both the correct and poisoned facts.
    """

    entity_id: str              
    entity_name: str            
    entity_type: str # "drug" or "disease"
    attribute: str 

    true_fact: str # one sentence stating the correct value
    false_fact: str  # one sentence stating a contradictory value

    question: str # the natural-language question asked at query time

    ground_truth_answer: str # short correct answer
    ground_truth_keywords: list[str] # variants used for keyword matching
    poison_answer: str  # short wrong answer
    poison_keywords: list[str]  #  variants of the wrong answer

    healthy_document: str # the ~80-word document that includes the truth 
    poisoned_document: str # thee ~80-word document including the "opposite"


@dataclass(frozen=True)
class Document:
    """
    Each MedicalEntity is split into two Documents: healthy and poisoned.
    """

    doc_id: str # identifies the unique document
    entity_id: str # link to the original MedicalEntity
    label: str # "healthy" or "poisoned"
    text: str # what's shown to the LLM


@dataclass(frozen=True)
class RetrievedContext:
    """
    The set of documents + query retrieved by the semantic search and
    used as the LLM's prompt.
    """

    query: str
    documents: list[Document]


@dataclass
class GenerationOutput:
    """
    The structured output from the LLM's raw response.
    """

    answer: str  
    conflict_detected: bool
    raw: str  # the untouched string the model produced
    parse_failed: bool # It's gonna be TRUE if the output format doesn't respect what we wanted
    reasoning: Optional[str] = None 


@dataclass
class QueryResult:
    """
    The complete record of a single RAG interaction, linking the retrieved 
    context to the generated output for a specific prompting strategy.
    """

    entity_id: str
    question: str
    strategy_name: str   # the prompting used

    context: RetrievedContext # the documents actually retrieved by RAG
    retrieval_complete: bool  # True if the search was accurate
    injection_used: bool  # True if forced injection was required

    generation: GenerationOutput


@dataclass
class EvalRecord:
    """
    The final deterministic evaluation for a single QueryResult.
    Categorizes the LLM's behavior into one of the four outcome buckets.
    """

    entity_id: str
    strategy_name: str
    outcome: str  # "correct" | "poisoned" | "flagged" | "other"

    ground_truth_hit: bool   # a keyword related to the truth was found in the answer
    poison_hit: bool  # a poisonous keyword was found in the answer
    lexicon_hit: bool   # a conflict-lexicon phrase was found (robustness cross-check)
    conflict_detected: bool 
    parse_failed: bool  # both coming from Generation Output too

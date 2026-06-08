"""
We create CorpusBuilder responsible for the loading the dataset generated and
checking its structural consistency with what we want (defined by 'MedicalEntity').
Then it separates the two context (true and false) (using 'Documents').
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.config import RAGConfig
from src.schema import Document, MedicalEntity


class CorpusBuilder:
    """
    Handles the transformation of raw JSON data into the Data type established.
    Guarantees deterministic document expansion and validates dataset integrity.
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self._entities: list[MedicalEntity] | None = None # populated by load()

    def load(self, path: Path | None = None) -> list[MedicalEntity]:
        """
        Reads the JSON corpus and maps it to MedicalEntity dataclasses.
        It will raise an error if the format is not
        """
        data_path = path if path is not None else self.config.data_path
        with open(data_path, "r", encoding="utf-8") as f:
            raw_entities = json.load(f)

        entities = []
        for index, raw in enumerate(raw_entities):
            try:
                entities.append(MedicalEntity(**raw))
            except TypeError as exc:
                entity_id = raw.get("entity_id", "<unknown>")
                raise ValueError(
                    f"Entity at index {index} (entity_id={entity_id}) does not "
                    f"match the MedicalEntity schema: {exc}"
                ) from exc

        self._entities = entities
        return entities

    def validate(self, entities: list[MedicalEntity] | None = None) -> None:
        """
        Enforces experimental integrity constraints before pipeline execution.
        Fails loud if any constraint is violated to prevent downstream evaluation errors.
        """
        entities = entities if entities is not None else self._entities
        if entities is None:
            raise ValueError("No entities loaded yet")

        seen_ids: set[str] = set()
        for entity in entities:
            # First we gotta ensure that entity_id is unique
            if entity.entity_id in seen_ids:
                raise ValueError(f"Duplicate entity_id found: {entity.entity_id!r}")
            seen_ids.add(entity.entity_id)

            # Second we ensure the keyword sets are completely disjoint 
            gt_keywords = {kw.lower() for kw in entity.ground_truth_keywords}
            poison_keywords = {kw.lower() for kw in entity.poison_keywords}
            overlap = gt_keywords & poison_keywords
            if overlap:
                raise ValueError(
                    f"{entity.entity_id}: ground_truth_keywords and "
                    f"poison_keywords overlap on {sorted(overlap)} — the "
                    f"evaluator would not be able to distinguish correct "
                    f"from poisoned answers for this entity."
                )

            # Third we ensure that the answers are not substrings of each other to prevent false positives
            gt_answer = entity.ground_truth_answer.lower()
            poison_answer = entity.poison_answer.lower()
            if gt_answer in poison_answer or poison_answer in gt_answer:
                raise ValueError(
                    f"{entity.entity_id}: ground_truth_answer "
                    f"({entity.ground_truth_answer!r}) and poison_answer "
                    f"({entity.poison_answer!r}) are not lexically disjoint "
                    f"— one is a substring of the other, which would make "
                    f"keyword matching ambiguous."
                )

    def build_documents(self, entities: list[MedicalEntity] | None = None) -> list[Document]:
        """
        Deterministically expands each MedicalEntity into a Healthy/Poisoned document pair.
        Doc IDs are derived in the same way for all documents (-H for healthy data, ID-P for poisoned data).
        """
        entities = entities if entities is not None else self._entities
        if entities is None:
            raise ValueError("No entities loaded yet — call load() first.")

        documents: list[Document] = []
        for entity in entities:
            documents.append(
                Document(
                    doc_id=f"{entity.entity_id}-H",
                    entity_id=entity.entity_id,
                    label="healthy",
                    text=entity.healthy_document, # the actual document
                )
            )
            documents.append(
                Document(
                    doc_id=f"{entity.entity_id}-P",
                    entity_id=entity.entity_id,
                    label="poisoned",
                    text=entity.poisoned_document, # the poisoned document
                )
            )
        return documents

    def get_queries(self, entities: list[MedicalEntity] | None = None) -> list[tuple[str, str]]:
        """
        Returns an ordered list of (entity_id, question) pairs to drive the execution loop.
        """
        entities = entities if entities is not None else self._entities
        if entities is None:
            raise ValueError("No entities loaded yet — call load() first.")

        return [(entity.entity_id, entity.question) for entity in entities]

    @staticmethod
    def hash_dataset(path: Path) -> str:
        """
        Computes a streamed SHA-256 hash of the JSON corpus.
        Essential for tracking dataset versioning and ensuring experimental reproducibility.
        """
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
"""
Orchestration module to run the RAG data poisoning experiment.

Executes Baseline and EpistemicVerification settings back-to-back
under identical conditions (same data, model, and vector index).
Outputs are saved in the results/ directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone

from src.config import RAGConfig
from src.corpus_builder import CorpusBuilder
from src.embedder import EmbeddingModel
from src.evaluator import Evaluator, MetricsReport
from src.llm import LLMClient
from src.prompts import (
    BaselineStrategy,
    EpistemicVerificationStrategy,
    PromptStrategy,
)
from src.rag_engine import RAGEngine
from src.schema import EvalRecord, MedicalEntity, QueryResult
from src.vector_store import ChromaStore

logger = logging.getLogger(__name__)

# Subset size for quick local pipeline checks).
SMOKE_ENTITY_LIMIT = 5


class ExperimentRunner:
    """Coordinates the data loading, indexing, and evaluation loops for both strategies."""

    def __init__(self, config: RAGConfig | None = None) -> None:
        self.config = config if config is not None else RAGConfig()

        # Light components are built immediately.
        self._builder = CorpusBuilder(self.config)
        self._evaluator = Evaluator()

        # Heavy components (embedder, vector store, LLM)
        # are delayed until run() to save memory
        self._store: ChromaStore | None = None
        self._llm: LLMClient | None = None

    def run(self, smoke: bool = False) -> None:
        """Runs the end-to-end experiment pipeline and saves metrics to disk."""
        # 1. Ingest dataset
        entities = self._builder.load()
        self._builder.validate(entities)

        if smoke:
            entities = entities[:SMOKE_ENTITY_LIMIT]
            print(f"[smoke] limited to first {len(entities)} entities")

        entity_by_id = {e.entity_id: e for e in entities}
        queries = self._builder.get_queries(entities)

        # 2. Build the vector index once
        self._index_corpus(entities)

        # # 3. Load LLM once and share it across settings to prevent memory bloat
        self._llm = LLMClient(self.config)

        # 4. Compare both prompt strategies under identical conditions
        strategies: list[PromptStrategy] = [
            BaselineStrategy(),
            EpistemicVerificationStrategy(),
        ]

        all_records: list[EvalRecord] = []
        all_log_rows: list[dict] = []
        reports: list[MetricsReport] = []

        for strategy in strategies:
            print(f"\n=== Running setting: {strategy.name} ===")
            engine = RAGEngine(self._store, self._llm, strategy, self.config)
            results = engine.run_batch(queries)

            # Evaluate generations against ground truth / poison keywords
            records = [
                self._evaluator.classify(r, entity_by_id[r.entity_id])
                for r in results
            ]
            report = self._evaluator.compute_metrics(records)

            all_records.extend(records)
            all_log_rows.extend(
                self._log_row(result, record)
                for result, record in zip(results, records)
            )
            reports.append(report)
            self._print_report(report)

        # 5. Persist detailed logs + aggregated metrics.
        self._save_records(all_log_rows, n_entities=len(entities), smoke=smoke)
        self._save_metrics(reports)


    def _index_corpus(self, entities: list[MedicalEntity]) -> None:
        """Populates the local vector store with the document corpus."""
        embedder = EmbeddingModel(self.config)
        self._store = ChromaStore(self.config, embedder)

        documents = self._builder.build_documents(entities)
        self._store.reset()          # drop any previous (possibly full-corpus) index
        self._store.add(documents)
        print(f"Indexed {len(documents)} documents for {len(entities)} entities.")

    # ------------------------------------------------------------------
    # Serialization helpers

    @staticmethod
    def _log_row(result: QueryResult, record: EvalRecord) -> dict:
        """Flattens tracking data into a single dictionary for logging."""
        gen = result.generation
        row = asdict(record)  # outcome + all the underlying boolean signals
        row.update(
            question=result.question,
            answer=gen.answer,
            reasoning=gen.reasoning,
            raw=gen.raw,
            injection_used=result.injection_used,
            retrieved_doc_ids=[d.doc_id for d in result.context.documents],
        )
        return row

    def _save_records(self, log_rows: list[dict], n_entities: int, smoke: bool) -> None:
        """Saves granular tracking records and configuration metadata to JSON."""
        self.config.results_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.config.results_dir / "experiment_records.json"

        payload = {
            "metadata": {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "smoke": smoke,
                "n_entities": n_entities,
                "n_records": len(log_rows),
                "model_id": self.config.model_id,
                "embed_id": self.config.embed_id,
                "top_k": self.config.top_k,
                "max_new_tokens": self.config.max_new_tokens,
                "seed": self.config.seed,
                "dataset_sha256": CorpusBuilder.hash_dataset(self.config.data_path),
            },
            "records": log_rows,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {len(log_rows)} records → {out_path}")

    def _save_metrics(self, reports: list[MetricsReport]) -> None:
        """Saves final aggregated metrics to a clean summary CSV."""
        self.config.results_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.config.results_dir / "experiment_metrics.csv"

        rows = [asdict(r) for r in reports]
        fieldnames = list(rows[0].keys())
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {len(rows)} metric rows → {out_path}")

    # ------------------------------------------------------------------
    @staticmethod
    def _print_report(report: MetricsReport) -> None:
        """Prints a snapshot of the outcome distributions to the console."""
        print(
            f"[{report.strategy_name}] N={report.n}  "
            f"correct={report.correct} poisoned={report.poisoned} "
            f"flagged={report.flagged} other={report.other}\n"
            f"    accuracy={report.factual_accuracy:.2f}  "
            f"poison_adoption={report.poison_adoption_rate:.2f}  "
            f"flag_rate={report.flag_rate:.2f}  "
            f"conflict_detection={report.conflict_detection_rate:.2f}  "
            f"parse_fail={report.parse_failure_rate:.2f}  "
            f"retrieval_recall={report.retrieval_recall_pair:.2f}"
        )


def main() -> None:
    """CLI entry point supporting smoke and full scale runs."""
    parser = argparse.ArgumentParser(
        description="Run the 'In RAG We Trust?' poisoning experiment "
        "(Baseline vs EpistemicVerification)."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"Fast verification: limit to the first {SMOKE_ENTITY_LIMIT} entities.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    ExperimentRunner().run(smoke=args.smoke)


if __name__ == "__main__":
    main()

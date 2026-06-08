"""
Configuration module for the experiment.
This is meant to assure the proper reproducibility 
and comparibility between the two prompting strategies
"""

from dataclasses import dataclass
from pathlib import Path

# Resolve the project root from this file's location
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True) # frozen makes the object immutable
class RAGConfig:
    """
    We must assure that the model, the retrievial, reproducibility
    is the same across both experiments (or rerunning the experiment)
    """

    # Which models we use 
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    embed_id: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Retrieval & generation knobs 
    top_k: int = 4 # documents retrieved per query
    max_new_tokens: int = 256 # cap on generation length

    # Reproducibility 
    # Seed for deterministic operations (e.g., embedding batch order).
    seed: int = 42

    #  Where things live on disk 
    data_path: Path = PROJECT_ROOT / "data" / "fictional_medicine_v1.json"
    chroma_dir: Path = PROJECT_ROOT / "data" / "chroma"
    results_dir: Path = PROJECT_ROOT / "results"

    # Name of the Chroma collection holding the 200 healthy/poisoned docs.
    collection_name: str = "fictional_medicine_v1"
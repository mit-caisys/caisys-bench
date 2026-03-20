"""
Directory path configuration for RAG experiments.
"""

from pathlib import Path

RAG_DIR = Path(__file__).resolve().parents[2]

DATASET_DIR = RAG_DIR / "dataset"
LOG_DIR = RAG_DIR / "log"
RAW_METRIC_DIR = RAG_DIR / "raw_metric"
PROCESSED_METRIC_DIR = RAG_DIR / "processed_metric"
METRIC_PLOT_DIR = RAG_DIR / "metric_plot"
POISSON_PLOT_DIR = RAG_DIR / "poisson_plot"
RESULT_PLOT_DIR = RAG_DIR / "result_plot"
RESULT_DIR = RAG_DIR / "result"
SEQUENTIAL_PLOT_DIR = RAG_DIR / "sequential_plot"
SRC_DIR = RAG_DIR / "src"
TIMELINE_DIR = RAG_DIR / "timeline"


MILVUS_PATH = DATASET_DIR / "milvus_vector_store.db"
EMBEDDING_CACHE_DIR = DATASET_DIR / "embedding_cache"
FRAMES_DIR = DATASET_DIR / "frames-benchmark"
FRAMES_PATH = FRAMES_DIR / "test.tsv"

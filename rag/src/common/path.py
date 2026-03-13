from pathlib import Path

RAG_DIR = Path(__file__).resolve().parents[2]

CONFIG_DIR = RAG_DIR / "config"
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

if __name__ == "__main__":
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RAW_METRIC_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_METRIC_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    POISSON_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    SEQUENTIAL_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    TIMELINE_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

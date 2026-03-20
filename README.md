# Compound AI Systems - Benchmark

This repository provides a complete setup for running various Compound AI workflows, e.g., RAG, and plotting the run results.
The inference backend mainly used by the workflows in this directory is `vLLM`.

---

## 📁 Repository Structure

```
├── deep_research/                  # DeepResearch workflow directory
├── load_generator/                 # Poisson request load generator directory
├── monitoring/                     # Metrics monitoring scripts, e.g., CPU/GPU utilization
├── openevolve/                     # OpenEvolve workflow directory
├── process_data/                   # Scripts for processing result of the run
├── rag/                            # RAG workflow directory
├── start_instance/                 # Base scripts for starting vllm instance
├── README.md                       # This file
└── install.sh                      # Installs dependencies
```

---

# RAG Benchmark

This repository provides a complete setup for running inference and plotting results on the [google/frames-benchmark](https://huggingface.co/datasets/google/frames-benchmark) dataset (with [wiki-links](https://code.google.com/archive/p/wiki-links/) as a large irrelevant dataset) using `vLLM` with Hugging Face models and OpenAI models.
The workflow is similar to [Self-RAG](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_self_rag.ipynb). 
You can choose whether to enable or disable each agent in your config file. 
However, most analysis scripts do not currently support evaluating those agents.

---

## 📁 Repository Structure

```
├── config/                         # Config repository
│   ├── config_openai.yaml          # Sample config file for using OpenAI 
│   └── config_vllm.yaml            # Sample config file for using vLLM 
├── dataset/                        # Dataset repository 
│   └── download.sh                 # Script for downloading google wikilinks
├── log/                            # Log repository (log of each workflow run)
├── metric_plot/                    # Plot of processed metric
├── poisson_plot/                   # Poisson specific plot 
├── processed_metric/               # Processed csv and json of raw metric 
├── raw_metric/                     # Raw metric log from dcgmi and sar
├── result/                         # Run result repository
├── result_plot/                    # Plot of all run results from plot_result script
├── sequential_plot/                # Sequential specific plot 
├── src/                            # Core source code repository
│   ├── common/
│   ├── self_rag/                   # Self-RAG workflow implementation
│   ├── combine_plots.py            # Script for creating a plot with relevant subplots
│   ├── create_large_vectorstore.py # Script for creating a large vectorstore using wikilinks
│   ├── grade.py                    # Script for grading the result in result repository
│   ├── plot_poisson.py             # Script for plotting poisson specific plot
│   ├── plot_result.py              # Wrapper for plotting data_vis script
│   ├── plot_sequential.py          # Script for plotting sequential specific plot
│   ├── process_and_plot_metric.py  # Wrapper for plot_metrics script
│   ├── run.sh                      # Wrapper for running the experiment (baseline / sequential / poisson)
│   ├── run_experiment.py           # Script for running the experiment (baseline / sequential / poisson)
│   ├── start_embedding.sh          # Wrapper for running vLLM embedding model
│   └── start_vllm.sh               # Wrapper for running vLLM inference model
├── timeline/                       # Timeline repository 
├── README.md                       # This file
├── install.sh                      # Installs dependencies and datasets 
└── requirements.txt                # Python dependencies
```

---

## 🛠 Installation

To set up the environment and install dependencies, run:

```bash
./install.sh
```

To create a large vectorstore, go to `dataset` directory and run:

```bash
python create_large_vectorstore.py
```

with `aadd_frames_data` once, and run with `acreate_database` until the database size is large enough.

Add environment variables to `comp-ai-bench/.env`:

```bash
OPENAI_API_KEY=sk-XXXXXXXXXX
HF_TOKEN=hf_XXXXXXXXXX
```

---

## 🚀 Running the Project (in `src`)

### 🧠 Run Inference

To use vLLM inference model, run the following script:

```bash
./start_vllm.sh
```

The default model is [google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it).
To change the model, modify the argument in `start_vllm.sh`.

To use vLLM embedding model, run the following script:

```bash
./start_embedding.sh
```

The default model is [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m).
To change the model, modify the argument in `start_embedding.sh`.

To start the experiment, run:

```bash
./run.sh [options]
```

The config file specified in `run.sh` will be used.
To run load generator experiment, pass `-l` flag with rate. For example,

```bash
./run.sh -l 0.1
```

will run the load experiment with poisson rate 0.1.

To grade the result, run:

```bash
python grade.py
```

### 📊 Plot Result/Metric

To plot the result, run:

```bash
python plot_result.py
```

To plot the metric, run:

```bash
python process_and_plot_metric.py
```

This script needs to be run before `plot_poisson.py`.

To plot sequential specific graph, run:

```bash
python plot_sequential.py
```

To plot poisson specific graph, run:

```bash
python plot_poisson.py
```

To plot a combine plot of timeline and metrics, run:

```bash
python combine_plots.py -m path/to/metrics/dir -t path/to/timeline.csv -o output/plot.pdf
```

---

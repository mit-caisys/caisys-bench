# RAG Benchmark

This repository provides a complete setup for running inference and plotting results on the [google/frames-benchmark](https://huggingface.co/datasets/google/frames-benchmark) dataset using `vLLM`, Hugging Face models and OpenAI models.

---

## 📁 Repository Structure

```
├── config/                         # Config repository
│   ├── config_huggingface.json     # Sample config file for using vLLM huggingface 
│   ├── config_ollama.json          # Sample config file for using Ollama (use vLLM if possible) 
│   └── config_openai.json          # Sample config file for using OpenAI 
├── dataset/                        # Dataset repository 
├── log/                            # Log repository (log of each workflow run)
├── metric_plot/                    # Plot of processed metric
├── poisson_plot/                   # Poisson specific plot 
├── processed_metric/               # Processed csv and json of raw metric 
├── raw_metric/                     # Raw metric log from dcgmi and sar
├── result/                         # Run result repository
├── result/                         # Plot of all run results from data_vis script
├── sequential_plot/                # Sequential specific plot 
├── src/                            # Core source code repository
│   ├── common/
│   ├── self_rag/                   # Self-RAG workflow implementation
│   ├── data_vis.py                 # Wrapper for data_vis script
│   ├── grade.py                    # Script for grading the result in result repository
│   ├── plot_poisson.py             # Script for plotting poisson specific plot
│   ├── plot_sequential.py          # Script for plotting sequential specific plot
│   ├── process_and_plot_metric.py  # Wrapper for plot_metrics script
│   ├── run_experiment.py           # Script for running the experiment (baseline / sequential / poisson)
│   ├── start_vllm.sh               # Wrapper for start_vllm.sh script
├── timeline/                       # Timeline repository 
└── README.md                       # This file
├── install.sh                      # Installs dependencies and datasets 
├── requirements.txt                # Python dependencies
```

---

## 🛠 Installation

To set up the environment and install dependencies, run:

```bash
./install.sh
```

Add environment variables to `src/.env`:

```bash
OPENAI_API_KEY=sk-XXXXXXXXXX
HF_TOKEN=hf_XXXXXXXXXX
```

---

## 🚀 Running the Project (in `src`)

### 🧠 Run Inference

To use vLLM huggingface model, run the following script:

```bash
./start_vllm.sh
```

The default model is [google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it).
To change the model, modify the argument in `start_vllm.sh`.

To start the experiment, run:

```bash
python run_experiment.py -f <config_file_in_config_repository> [options]
```

If no config file is passed, `CONFIG_PATH` varible at the top of `run_experiment.py` will be used.
To run load generator experiment, pass `-l` flag with rate. For example,

```bash
python run_experiment.py -l 0.1
```

will run the load experiment with poisson rate 0.1.

To grade the result, run:

```bash
python grade.py
```

### 📊 Plot Result/Metric

To plot the result, run:

```bash
python data_vis.py
```

To plot sequential specific graph, run:

```bash
python plot_sequential.py
```

To plot poisson specific graph, first run:

```bash
python process_and_plot_metric.py
```

then run:

```bash
python plot_poisson.py
```

---

## 🔍 Findings & Future Improvements

### 📌 Key Findings

- The findings are summarized in [the paper](https://www.overleaf.com/project/6880fb23b0dff674cc66765e).

### 🧠 Ideas for Improvement

- [ ] Hallucination grader and answer grader are not used effectively. For complicated questions, these graders are incorrect most of the time.
- [ ] Use Chain-of-Thought with rag to retrieved more relevant documents for subquestions.
- [ ] Use different models for each LLM agents.

---

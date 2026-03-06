# Deep Research Benchmark

This repository provides a complete setup for running inference and plotting results for [Smolagents Open Deep Research](https://github.com/huggingface/smolagents/tree/main/examples/open_deep_research) workflow using `vLLM` with Hugging Face models.

---

## 📁 Repository Structure

```
├── agent_log/                      # Agent log repository (CodingAgent console logs)
├── configs/                        # Config repository
│   └── config.yaml                 # Sample config file
├── metric_plot/                    # Plot of processed metric
├── processed_metric/               # Processed csv and json of raw metric 
├── raw_metric/                     # Raw metric log from dcgmi and sar
├── scripts/                        # Tools repository 
├── step_log/                       # Step log repository (CodingAgent steps)
├── README.md                       # This file
├── install.sh                      # Installs dependencies
├── path.py                         # Path specificaion
├── process_and_plot_metric.py      # Script to process and plot metric (called after the run is done, i.e., raw_metric/ is created)
├── requirements.txt                # Python dependencies
├── run.py                          # Entry point for Deep Research (called from run.sh)
├── run.sh                          # Run for each config in config folder
└── start_vllm.sh                   # Script to start vllm instance 
```

---

## 🛠 Installation

To set up the environment and install dependencies, run:

```bash
./install.sh
```

Add environment variables to `comp-ai-bench/.env`:

```bash
HF_TOKEN=hf_XXXXXXXXXX
SERPAPI_API_KEY=XXXXXXXXXXXXX
```

---

## 🚀 Running the Project

### 🧠 Run Inference

To use vLLM inference model, run the following script:

```bash
./start_vllm.sh
```

The default model is [google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it).
To change the model, modify the argument in `start_vllm.sh`.

To start the experiment, run:

```bash
./run.sh [options]
```

To specify the question to run, pass `-i` flag with the question index. For example

```bash
./run.sh -i 1
```

will run the second question in each config file (default index is 0).

### 📊 Plot Result/Metric

To plot the metric, run:

```bash
python process_and_plot_metric.py
```

---

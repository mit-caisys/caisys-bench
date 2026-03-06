# OpenEvolve Benchmark

This repository provides a complete setup for running inference and plotting results for [openevolve](https://github.com/algorithmicsuperintelligence/openevolve) workflow with [circle packing](https://github.com/algorithmicsuperintelligence/openevolve/tree/main/examples/circle_packing) and [rust adaptive sorting](https://github.com/algorithmicsuperintelligence/openevolve/tree/main/examples/rust_adaptive_sort) examples using `vLLM`, Hugging Face models.

---

## 📁 Repository Structure

```
├── circle_packing/                             # Circle packing repository
│   ├── src/      
│   │   ├── configs/                            # Config repository
│   │   │   └── config.yaml                     # Config file 
│   │   ├── templates/                          # Template directory (use if template_dir is specified in the config) 
│   │   │   ├── diff_default.txt                # Default diff prompt (unused)
│   │   │   ├── diff_user.txt                   # Current diff prompt
│   │   │   ├── full_rewrite_default.txt        # Default full_rewrite prompt (unused)
│   │   │   └── full_rewrite_user.txt           # Current full_rewrite prompt
│   │   ├── evaluator.py                        # Evaluator for circle packing 
│   │   ├── initial_program.py                  # Initial circle packing program 
│   │   └── oracle.json                         # Cached evaluated results
│   └── run.sh                                  # Run for each config in config folder
├── rust_adaptive_sort/                         # Rust adaptive sorting repository
│   ├── src/      
│   │   ├── configs/                            # Config repository
│   │   │   └── config.yaml                     # Config file 
│   │   ├── sort_test/                          # Repository used for evaluation 
│   │   ├── templates/                          # Template directory (use if template_dir is specified in the config) 
│   │   │   ├── diff_default.txt                # Default diff prompt (unused)
│   │   │   ├── diff_user.txt                   # Current diff prompt
│   │   │   ├── evolution_history_default.txt   # Default evolution history prompt (unused)
│   │   │   ├── evolution_history.txt           # Current evolution history prompt
│   │   │   ├── full_rewrite_default.txt        # Default full_rewrite prompt (unused)
│   │   │   └── full_rewrite_user.txt           # Current full_rewrite prompt
│   │   ├── evaluator.py                        # Evaluator for circle packing 
│   │   ├── initial_program.rs                  # Initial circle packing program 
│   │   └── oracle.json                         # Cached evaluated results
│   └── run.sh                                  # Run for each config in config folder
├── README.md                                   # This file
├── analyze_frequencies.py                      # Library for analyzing input program frequencies
├── calculate_energy.py                         # Script for calculating energy at given time offset
├── combine_cache_plots.py                      # Script to combine KV cache hit rate and lifetime plots
├── combine_score_plots.py                      # Script to combine score evolution plots
├── install.sh                                  # Installs dependencies and datasets 
├── iteration_scores.py                         # Library for processing per-iteration scores
├── log_parser.py                               # Library for parsing OpenEvolve logs
├── openevolve-run.py                           # Entry point for OpenEvolve (called from run.sh)
├── paths.py                                    # Path configuration
├── plot_activity.py                            # Library for plotting activity timelines
├── plot_scores.py                              # Library for plotting score evolution
├── process_and_plot_metric.py                  # Script to process and plot metric (called after the run is done)
├── requirements.txt                            # Python dependencies
├── start_vllm.sh                               # Script to start vllm instance 
└──
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
```

---

## Update to OpenEvolve

### Sampler Update

In `config.py`, inside the `PromptConfig` class, add `random_seed`:

```bash
@dataclass
class PromptConfig:
    ...
    random_seed: Optional[int] = 42
    ...
```

In `prompt/sampler.py`, inside the `__init__` function, add `random_seed` condition:

```bash
class PromptSampler:
    ...
    def __init__(self, config: PromptConfig):
        ...
        if config.random_seed is not None:
            random.seed(config.random_seed)
            logger.debug(f"Sampler: Set random seed to {config.random_seed}")
    ...
```

### Database Update

In config, inside `database` section, add `batch_per_island`:

```bash
database:
    ...
    batch_per_island: 1
    ...
```

In `config.py`, inside the `DatabaseConfig` class, add `batch_per_island`:

```bash
@dataclass
class DatabaseConfig:
    ...
    batch_per_island: Optional[int] = None
    ...
```

In `process_parallel.py`, modify `batch_per_island` calculation:

```bash
async def run_evolution(...):
    ...
    batch_per_island = (
        self.config.database.batch_per_island
        or max(1, batch_size // self.num_islands)
        if batch_size > 0
        else 0
    )
    ...
```

In `database.py`, change all uses of set for `self.islands` and `self.archive` to `OrderSet`.

### Prompt Reordering Update

In config, inside `prompt` section, add `sort_diverse_programs`:

```bash
prompt:
    ...
    sort_diverse_programs: true
    ...
```

In `config.py`, inside the `PromptConfig` class, add `sort_diverse_programs`:

```bash
@dataclass
class PromptConfig:
    ...
    sort_diverse_programs: bool = False
    ...
```

In `prompt/sampler.py`, inside `_format_evolution_history` function, add the sorting logic:

```bash
def _format_evolution_history(...):
    ...
    if num_diverse > 0
        diverse_programs = random.sample(remaining_programs, num_diverse)
        if self.config.sort_diverse_programs:
            diverse_programs.sort(key=remaining_programs.index)
    ...
```

In config, inside `database` section, add `sort_inspiration_programs`:

```bash
database:
    ...
    sort_inspiration_programs: true
    ...
```

In `config.py`, inside the `DatabaseConfig` class, add `sort_inspiration_programs`:

```bash
@dataclass
class DatabaseConfig:
    ...
    sort_inspiration_programs: bool = False
    ...
```

In `database.py`, inside `sample_from_island` function, add the sorting logic:

```bash
def sample_from_island(...):
    ...
    else:
        inspiration_ids = random.sample(other_programs, num_inspirations)
        if self.config.sort_inspiration_programs:
            inspiration_ids.sort(key=other_programs.index)
    ...
```

---

## 🚀 Running the Project

### 🧠 Run Inference

To use vLLM huggingface model, run the following script:

```bash
./start_vllm.sh
```

To change the model, modify the argument in `start_vllm.sh`.
Make sure that TENSOR_PARALLEL_SIZE matches with `run.sh` and the model matches with `config.yaml`.

To start the experiment, go to respective workflow directory and run:

```bash
./run.sh
```

The experiment will run for each config fil in `src/configs` directory.

### 📊 Plot Result/Metric

To process and plot the result, within the workflow directory, run:

```bash
python ../process_and_plot_metric.py 
```

There are special plotting scripts in `openevolve/` directory that can be run by passing corresponding arguments.

---

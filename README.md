# Benchmarking Compound AI Applications for Hardware-Software Co-Design

This repository provides a complete setup for running various Compound AI workflows, e.g., RAG, and plotting the run results.
The inference backend mainly used by the workflows in this directory is `vLLM`.

This is an artifact of our ongoing project. A pre-print of our paper can be found [here](https://arxiv.org/pdf/2604.09593).

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

## Launching Experiment

To run an experiment, navigate to the desired [workflow](#workflows) and complete the installation steps. Then, follow the instructions in the **Running the Project** section.

---

## Workflows

### RAG

An RAG workflow is designed to evaluate retrieval and question answering on the [frames](https://huggingface.co/datasets/google/frames-benchmark) benchmark dataset.
The implementation uses [langgraph](https://github.com/langchain-ai/langgraph) library and is adapted from a [Self-RAG](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_self_rag.ipynb) example.

For more details, see [rag](rag/README.md).

### OpenEvolve

An OpenEvolve workflow does iterative improvement on optimizing [circle packing](https://github.com/algorithmicsuperintelligence/openevolve/tree/main/examples/circle_packing) algorithms and [rust adaptive sorting](https://github.com/algorithmicsuperintelligence/openevolve/tree/main/examples/rust_adaptive_sort) algorithms.
The implementation uses [openevolve](https://github.com/algorithmicsuperintelligence/openevolve) library and are adapted from [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve/main/examples) examples.

For more details, see [openevolve](openevolve/README.md).

### Deep Research

A Deep Research workflow performs question answering by interacting with a web browser to gather information and respond to user queries.
The implementation uses [smolagents](https://github.com/huggingface/smolagents) library and is adapted from [Open Deep Research](https://github.com/huggingface/smolagents/tree/main/examples/open_deep_research) example.

For more details, see [deep_research](deep_research/README.md).

---

## Contact

If you have suggestions or improvements to this work, please feel free to open a pull-request.
Otherwise, you may also contact us at: [Paramuth Samuthrsindh](mailto:paramuth@mit.edu) or [Angel Cervantes](mailto:cerangel@mit.edu).

## Citation

If you use this codebase, or otherwise found our work valuable, please cite:
```
@misc{samuthrsindh2026benchmarkingcompoundaiapplications,
      title={Benchmarking Compound AI Applications for Hardware-Software Co-Design}, 
      author={Paramuth Samuthrsindh and Angel Cervantes and Varun Gohil and Gohar Irfan Chaudhry and Christina Delimitrou and Adam Belay},
      year={2026},
      eprint={2604.09593},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2604.09593}, 
}
```

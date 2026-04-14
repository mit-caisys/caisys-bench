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

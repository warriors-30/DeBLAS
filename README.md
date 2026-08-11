# DeBLAS: Accelerate LLM Pretraining by Length-based Sequence Scheduling


## Install experiment dependencies
```
pip install -r requirements.txt
```

## Quick Start
The training script is in `scripts/benchmark_c4`.

Pretraining a LLaMA 130M model on C4 with random batch selection
```train
CUDA_VISIBLE_DEVICES=0 bash scripts/benchmark_c4/llama_130m.sh
```

Pretraining a LLaMA 130M model on C4 with DeBLAS
```train
CUDA_VISIBLE_DEVICES=0 bash scripts/benchmark_c4/llama_130m_deblas.sh
```

## Acknowledgement

We appreciate the following papers for their open-source code, which this repository is built upon.
* [ReloRA: High-rank training through low-rank updates. (ICLR 2024)](https://github.com/guitaricet/relora).
* [GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection. (ICML 2024)](https://github.com/jiaweizzhao/galore).
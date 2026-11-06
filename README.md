# Beyond Historical Similarity: Self-Error-Aware Retrieval for Time Series Forecasting

This repository contains the anonymous implementation of SE-RAG, a retrieval-augmented framework for zero-shot time series forecasting.

## Overview

Retrieval can supplement a frozen time series foundation model with historical contexts and their future segments. However, retrieved candidates may be incomplete or noisy, and historical similarity alone does not reliably indicate whether a candidate future is useful for the query.

The method preserves the forecasting interface of the underlying backbone and supports both Chronos-Bolt and MOMENT.

## Repository Structure

```text
.
├── TS-RAG/
│   ├── models/              # Forecasting backbones and retrieval fusion modules
│   ├── script/              # Training and zero-shot evaluation scripts
│   ├── pretrain.py          # Retrieval-module pretraining entry point
│   └── zeroshot.py          # Zero-shot forecasting entry point
├── requirements.txt
└── README.md
```

## Zero-Shot Evaluation

Before running an experiment, configure the dataset, retrieval database, text-retrieval embeddings, self-error features, checkpoint path, and CUDA device in the corresponding script.

## Experimental Setting

Experiments follow a zero-shot protocol with a context length of 512 and forecasting horizon of 64. Evaluation covers ETTh1, ETTh2, ETTm1, ETTm2, Weather, and Exchange Rate using MSE and MAE.

## License

This code is provided for research purposes.

# Pyper: Tiny MoEs

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

## Contents

1. [Overview](#overview)
2. [Model Architecture](#model-architecture)
3. [Training](#training)
4. [Requirements](#requirements)
5. [References](#references)

## Overview

This repository contains the implementation of a Mixture of Experts (MoE) model, featuring 8 individual transformer experts. Each expert has approximately 1.5 billion parameters, culminating in a total of around 12 billion parameters for the entire model. The model is designed for high-performance applications, incorporating advanced transformer structures and training methodologies.

## Model Architecture

The MoE model consists of 8 transformer experts, each with key components:

- **Root Mean Square Layer Normalization (RMSNorm)**
- **Rotary Positional Encoding (RoPE)**
- **FeedForward Neural Network**

### Transformer Components:

- **RMSNorm**: A normalization layer that uses the root mean square value of the input tensor.
- **RoPE**: Implements Rotary Positional Encoding in the attention mechanism to encode sequence order information.
- **FeedForward**: Comprises two linear transformations with a GELU activation function.

### Transformer Block:

Each block within the transformer model includes:

- A multi-head self-attention mechanism with RoPE.
- A feedforward neural network.
- Residual connections and layer normalization.

## Training

The training of each expert employs a custom training loop, including loss computation and backpropagation. The training procedure is adaptable to various datasets and training regimes.

### Key Training Parameters:

- **Learning Rate**
- **Batch Size**
- **Number of Epochs**
- **Early Stopping**
- **Warmup Scheduling**

## Requirements

- Python 3.1
- PyTorch
- einops

## References

1. [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
2. [Mixture-of-Experts with Expert Choice Routing](https://arxiv.org/abs/2202.09368)

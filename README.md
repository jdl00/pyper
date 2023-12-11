# Pyper: Selective State Spaces with Expert Choice Routing

## Contents

1. [About](#about)
1. [Aims](#aims)


## About

This project aims to build on the paper: [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) by creating a mixture of expert model using Google's [
Mixture-of-Experts with Expert Choice Routing](https://blog.research.google/2022/11/mixture-of-experts-with-expert-choice.html?m=1). The aim is to create a large model capable of generating natural language, while maintaing scalability across the model.

## Aims
- Use 4 experts within the mixture of expert model.
- Fine tune the Mamba models on 4 specific *expert* areas. *(see below)*
- The whole model should maintain Average-Case Linear Complexity


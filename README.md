# Pyper: Small MoEs

## Contents

1. [About](#about)
2. [Aims](#aims)
3. [Experts](#experts)


## About

This project aims to build on Google's [Mixture-of-Experts with Expert Choice Routing](https://blog.research.google/2022/11/mixture-of-experts-with-expert-choice.html?m=1). The model is Transformer which uses Rotary Position Embedding, these models are trained in domain specific areas.

## Aims

- Use 4 experts within the mixture of expert model.
- Fine tune the Transformer models on 4 specific *expert* areas. *(see below)*

## Experts

1. **Technical Domain Knowledge:** Dataset containing the code training examples.

2. **Natural Language Understanding:** Focuses on understanding and interpreting natural language questions.

3. **Problem Solving and Logic:** Focuses on problem-solving, logic formulation, and algorithmic thinking.

4. **General Knowledge:** Focuses on general knowledge for understand questions

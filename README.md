# Multi-Layer Perceptron — Maturaarbeit

From-scratch implementation of a fully-connected neural network in Python/NumPy, developed as part of the Swiss Matura thesis (Gymnasium).

## Overview

Implementation of a multi-layer perceptron without deep learning frameworks — forward pass, backpropagation, and gradient descent derived and coded from first principles.

## What is implemented

- Forward pass with configurable depth and width
- Backpropagation via chain rule (manual, no autograd)
- Activation functions: ReLU, sigmoid, tanh
- Loss functions: MSE, cross-entropy
- Optimizers: SGD, SGD with momentum
- Weight initialization: random, Xavier
- Evaluation on MNIST

## Purpose

This project predates the other projects in this portfolio and served as the foundation for understanding how neural networks work before moving to framework-based implementations (PyTorch). Deriving and implementing backpropagation manually remains one of the most direct ways to understand what deep learning frameworks abstract away.

## Setup

```bash
pip install numpy matplotlib
python mlp.py
```

No deep learning frameworks required — NumPy only.

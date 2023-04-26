# Liquid Time-stochasticity Networks (LTSs)

[![CodeQL](https://github.com/Ammar-Raneez/FYP_Algorithm/actions/workflows/codeql.yml/badge.svg)](https://github.com/Ammar-Raneez/FYP_Algorithm/actions/workflows/codeql.yml)
[![CodeFactor](https://www.codefactor.io/repository/github/ammar-raneez/liquid-time-stochasticity-networks/badge)](https://www.codefactor.io/repository/github/ammar-raneez/liquid-time-stochasticity-networks)

This is the official repository for Liquid TIme-stochasticity networks described in paper: https://doi.org/10.1109/CCWC57344.2023.10099071 

This implementation utilizes the Euler Maruyama solver to perform forward propagation and relies on the conventional backpropagation through-time (BPTT) to train the models.

## Prerequisites

The architecture was built using Keras and TensorFlow 2.0+ and Python 3+ on the Windows 11 machine.

## Experiments

`experiments/experiments.ipynb` demonstrates a couple of experiments attempting to model bitcoin prices.

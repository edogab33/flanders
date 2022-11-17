# FLANDERS (WIP!)

## Why
This repository is part of the work done for FLANDERS (read the abstract below) and serves as a way to reproduce and validate the results presented in the paper, but also useful to the Federated Learning community to quickly setup simulations in Flower with byzantine clients and return metrics. The contributions that this repository provides are listed below:

1. Implementation of the following defence baselines:
    - FedMedian
    - TrimmedMean
    - Krum
    - MultiKrum
    - Bulyan
    - FLTrust
    - FLANDERS

2. Implementation of the following attacks:
    - Gaussian Attack
    - LIE Attack
    - Fang Attack
    - MinMax Attack

[Add citations]

## Abstract
In this work, we propose FLANDERS, a novel federated learning (FL) aggregation scheme robust to Byzantine attacks.
FLANDERS considers the local model updates sent by clients at each FL round as a matrix-valued time series. Then, it identifies malicious clients as outliers of this time series by comparing actual observations with those estimated by a matrix autoregressive forecasting model. 
Experiments conducted on several datasets under different FL settings demonstrate that FLANDERS can mitigate more significantly than state-of-the-art baselines the detrimental effect of Byzantine attacks on the predictive accuracy of the global model without sacrificing its convergence time.

# Run

## Preliminaries
To run the code is advised to create a conda environment and install all the dependences contained in environment.yml.
[...]

## Configuration
[...]

# Authors
Dimitri Belli, Edoardo Gabrielli, Gabriele Tolomei

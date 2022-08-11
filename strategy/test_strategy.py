import numpy as np
from strategy.fedmscred import FedMSCRED
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

params = np.load("/Users/eddie/Documents/Università/ComputerScience/Thesis/flwr-pytorch/strategy/results.npy", allow_pickle=True)
print(params.shape)

strat = FedMSCRED()

strat.aggregate_fit(1, [(0, params[0]), (1, params[1])], [])
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

'''
Run this script to debug code errors inside new strategies without calculating the weights of 
the clients.
'''

params = np.load("strategy/results.npy", 
    allow_pickle=True)
print(params.shape)

strat = FedMSCRED()

strat.aggregate_fit(1, [(0, params[0]), (1, params[1])], [])
import numpy as np
import json
from strategy.fedmscred import FedMSCRED
from strategy.fedmedian import FedMedian
from strategy.krum import Krum
from strategy.multikrum import MultiKrum
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
    Status,
    Code
)

'''
Run this script to debug code errors inside new strategies without calculating the weights of 
the clients.
'''
params = np.load("strategy/test_params/weights_results.npy", allow_pickle=True)

print(params.shape)

strat = Krum()
params = [FitRes(parameters=params[i], num_examples=i, status=Status(code=Code.OK, message=''), metrics={}) for i in range(params.shape[0])]
strat.aggregate_fit(1, [(0, params[0]), (1, params[1])], [])
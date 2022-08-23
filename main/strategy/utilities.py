import numpy as np
import os
from typing import Dict, Optional, Tuple
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

def save_params(parameters_aggregated: Parameters, server_round: int):
    # TODO: test that:
    if parameters_aggregated is not None:
        # Save weights
        print("Loading previous weights...")
        # check file exists
        if os.path.isfile("strategy/util_histories/results.npy"):
            old_params = np.load("strategy/util_histories/weights.npy")
        else:
            old_params = np.array([])
        print(f"Saving round "+ str(server_round) +" weights...")
        new_params = np.vstack((old_params, parameters_aggregated.tensors))
        print("new param: "+str(new_params.shape))
        np.save(f"strategy/util_histories/round-"+str(server_round)+"-weights.npy", new_params)


def evaluate_aggregated(
    evaluate_fn, server_round: int, parameters: Parameters, config: Dict[str, Scalar]
) -> Optional[Tuple[float, Dict[str, Scalar]]]:
    """Evaluate model parameters using an evaluation function."""
    if evaluate_fn is None:
        # No evaluation function provided
        return None
    parameters_ndarrays = parameters_to_ndarrays(parameters)
    eval_res = evaluate_fn(server_round, parameters_ndarrays, config)
    if eval_res is None:
        return None
    loss, metrics = eval_res
    return loss, metrics
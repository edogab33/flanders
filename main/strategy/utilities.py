import numpy as np
import os
import pandas as pd
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

def save_history_average(weights_results):
    params = np.asarray([])
    for par, _ in weights_results:
        flattened_params = np.concatenate([w.flatten() for w in par])
        print(np.mean(flattened_params))
        params = np.append(params, np.mean(flattened_params))

    # check that strategy/histoies directory exists and load history if it does
    history = np.load("strategy/histories/history.npy") if os.path.exists("strategy/histories/history.npy") else np.array([])
    history = np.vstack((history, params)) if history.size else params
    np.save("strategy/histories/history.npy", history)
    
    df = pd.DataFrame(history.T)
    df.to_csv("strategy/histories/history.csv", index=False, header=False)

def save_history_average_diff(weights_results):
    params = np.asarray([])
    for par, _ in weights_results:
        flattened_params = np.concatenate([w.flatten() for w in par])
        print("avg:")
        print(np.mean(flattened_params))
        params = np.append(params, np.mean(flattened_params))

    # check that strategy/histoies directory exists and load history if it does
    history = np.load("strategy/histories/history.npy") if os.path.exists("strategy/histories/history.npy") else np.array([])
    history = np.vstack((history, np.subtract(history[-1], params))) if history.size else params
    print("difference of avg:")
    print(history[-1])
    np.save("strategy/histories/history.npy", history)
    
    df = pd.DataFrame(history.T)
    df.to_csv("strategy/histories/history.csv", index=False, header=False)
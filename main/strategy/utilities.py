from locale import normalize
import string
import numpy as np
import os
import pandas as pd
from typing import Dict, Optional, Tuple, List
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

def save_params(parameters, cid, remove_last=False):
        new_params = parameters
        # Save parameters in client_params/cid_params
        path = f"clients_params/{cid}_params.npy"
        if os.path.exists(path):
            # load old parameters
            old_params = np.load(path, allow_pickle=True)
            if remove_last:
                old_params = old_params[:-1]
            # add new parameters
            new_params = np.vstack((old_params, new_params))
        # save parameters
        np.save(path, new_params)

def load_all_time_series(dir=""):
        """
        Load all time series in order to have a tensor of shape (m,T,n)
        where:
        - T := time;
        - m := number of clients;
        - n := number of parameters
        """
        files = os.listdir(dir)
        files.sort()
        data = []
        for file in files:
            data.append(np.load(os.path.join(dir, file), allow_pickle=True))
        return np.array(data)

def load_time_series(dir="", cid=0):
    """
    Load time series of client cid in order to have a matrix of shape (T,n)
    where:
    - T := time;
    - n := number of parameters
    """
    files = os.listdir(dir)
    files.sort()
    data = []
    for file in files:
        if file == f"{cid}.npy":
            data.append(np.load(os.path.join(dir, file), allow_pickle=True))
    return np.array(data)

def flatten_params(params):
    """
    Transform a list of parameters into a single vector of shape (n).
    """
    params_flattened = []
    for i in range(len(params)):
        params_flattened.append([])
        for j in range(len(params[i])):
            p = np.hstack(params[i][j])
            for k in range(len(p)):
                params_flattened[i].append(p[k])
    return np.array(params_flattened)

def update_confusion_matrix(
    cm:List[List[int]], 
    ground_truth:Dict[str, bool], 
    predicted_as_false: List[int], 
    predicted_as_true: List[int]
) -> List[List[int]]:
    """
    cm := [
        [TP, FP]
        [FN, TN]
    ]

    ground_truth := dictonary of {cid:malicious}
    predicted_as_false := list of cids predicted as false
    predicted_as_true := list of cids predicted as true
    """
    for cid, label in ground_truth.items():
        if label == True:
            if int(cid) in predicted_as_true:
                cm[0][0] += 1                           # TP
            elif predicted_as_false:
                cm[1][0] += 1                           # FN
            else:
                print("Error: ground truth is true but client is not predicted as true or false")
        elif label == False:
            if int(cid) in predicted_as_true:
                cm[0][1] += 1                           # FP
            elif predicted_as_false:
                cm[1][1] += 1                           # TN
            else:
                print("Error: ground truth is false but client is not predicted as true or false")
        else:
            print("Error: ground truth is not true or false")
    return cm


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
    """
    Compress the weights into a single vector by averaging and save it to a file.
    """
    params = np.asarray([])
    for par, _ in weights_results:
        flattened_params = np.concatenate([w.flatten() for w in par])
        params = np.append(params, np.mean(flattened_params))

    # check that strategy/histoies directory exists and load history if it does
    history = np.load("strategy/histories/history_avg.npy") if os.path.exists("strategy/histories/history_avg.npy") else np.array([])
    history = np.vstack((history, params)) if history.size else params
    np.save("strategy/histories/history_avg.npy", history)
    
    df = pd.DataFrame(history.T)
    df.to_csv("strategy/histories/history_avg.csv", index=False, header=False)

def save_history_avergage_normalized(weights_results):
    """
    Compress the weights into a single vector by normalizing values 
    within the same layer averaging and save it to a file.
    """
    params = np.asarray([])
    for par, _ in weights_results:
        # min-max normalization within the same layer
        normalized_params = [(layer)/(np.max(layer)) for layer in par]
        flattened_params = np.concatenate([w.flatten() for w in normalized_params])
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

def save_history_stack(weights_results, round):
    """
    Stack the weights into a single vector for each client and save it to a file.
    """
    params = np.asarray([])
    for par, _ in weights_results:
        flattened_params = np.concatenate([w.flatten() for w in par])
        params = np.vstack((params, flattened_params)) if params.size else flattened_params

    np.save("strategy/histories/history.npy", params)
    
    df = pd.DataFrame(params.T)
    df.to_csv("strategy/histories/history_stack_r"+str(round)+".csv", index=False, header=False)

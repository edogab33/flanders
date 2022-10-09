import numpy as np
import os
import json
import pandas as pd
from natsort import natsorted
from typing import Dict, Optional, Tuple, List
from flwr.common import (
    Parameters,
    Scalar,
    parameters_to_ndarrays,
)

def save_params(parameters, cid, remove_last=False, rrl=False):
    """
    Args:
    - parameters (ndarray): decoded parameters to append at the end of the file
    - cid (int): identifier of the client
    - remove_last (bool): if True, remove the last saved parameters and replace with "parameters"
    - rrl (bool): if True, remove the last saved parameters and replace with the ones saved before this round
    """
    new_params = parameters
    # Clip parameters
    new_params[new_params > 1e+3] = 1e+3
    # Save parameters in client_params/cid_params
    path = f"clients_params/{cid}_params.npy"
    if os.path.exists("clients_params") == False:
        os.mkdir("clients_params")
    if os.path.exists(path):
        # load old parameters
        old_params = np.load(path, allow_pickle=True)
        if remove_last:
            old_params = old_params[:-1]
            if rrl:
                new_params = old_params[-1]
        # add new parameters
        new_params = np.vstack((old_params, new_params))
    # save parameters
    np.save(path, new_params)

def save_results(loss, accuracy, config=None):
    # Save results as npy file
    dirs = [f for f in os.listdir("results_graphs/") if not f.startswith('.')]
    longest_string = len(max(dirs, key=len))
    idx = -2 if longest_string > 5 else -1

    highest_number = str(max([int(x[idx:]) for x in dirs if x[idx:].isdigit()]))
    loss_series = []
    acc_series = []
    loss_path = "results_graphs/run_"+highest_number+"/loss.npy"
    acc_path = "results_graphs/run_"+highest_number+"/acc.npy"
    if os.path.exists(loss_path):
        loss_series = np.load(loss_path)
    if os.path.exists(acc_path):
        acc_series = np.load(acc_path)
    loss_series = np.save(loss_path, np.append(loss_series, loss))
    acc_series = np.save(acc_path, np.append(acc_series, accuracy))

    # Save config
    config_path = "results_graphs/run_"+highest_number+"/config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)

    # Generate csv
    config["accuracy"] = accuracy
    config["loss"] = loss
    df = pd.DataFrame.from_records([config])
    csv_path = "results_all/all_results.csv"
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)

def load_all_time_series(dir="", window=0):
        """
        Load all time series in order to have a tensor of shape (m,T,n)
        where:
        - T := time;
        - m := number of clients;
        - n := number of parameters
        """
        files = os.listdir(dir)
        files = natsorted(files)
        data = []
        for file in files:
            data.append(np.load(os.path.join(dir, file), allow_pickle=True))
        return np.array(data)[:,-window:,:]

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
        if file == f"{cid}_params.npy":
            data = np.load(os.path.join(dir, file), allow_pickle=True)
    return np.array(data)

def flatten_params(params):
    """
    Transform a list of (layers-)parameters into a single vector of shape (n).
    """
    return np.concatenate(params, axis=None).ravel()

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

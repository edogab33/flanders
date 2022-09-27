import math
import numpy as np
from typing import Dict, List, Tuple
from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    FitRes,
    NDArrays,
    Parameters,
)
from flwr.server.client_proxy import ClientProxy
from scipy.stats import norm

def no_attack(
        ordered_results:List[Tuple[ClientProxy, FitRes]], 
        states:Dict[str, bool], 
        **kwargs
    ) -> List[Tuple[ClientProxy, FitRes]]:
    return ordered_results, {}

def gaussian_attack(
        ordered_results:List[Tuple[ClientProxy, FitRes]], 
        states:Dict[str, bool], 
        **kwargs
    ) -> List[Tuple[ClientProxy, FitRes]]:
    magnitude = kwargs.get("magnitude", 0.0)
    results = ordered_results.copy()
    for proxy, fitres in ordered_results:
        if states[fitres.metrics["cid"]]:
            params = parameters_to_ndarrays(fitres.parameters)
            perturbate = lambda a: a + np.random.normal(loc=0, scale=magnitude, size=len(a))
            new_params = np.apply_along_axis(perturbate, 0, params)
            fitres.parameters = ndarrays_to_parameters(new_params)
            results[int(fitres.metrics['cid'])] = (proxy, fitres)
    return results, {}

def lie_attack(
        ordered_results:List[Tuple[ClientProxy, FitRes]], 
        states:Dict[str, bool],
        **kwargs
    ) -> List[Tuple[ClientProxy, FitRes]]:
    """
    Omniscent LIE attack, Baruch et al. (2019)
    """
    results = ordered_results.copy()
    params = [parameters_to_ndarrays(fitres.parameters) for _, fitres in results]
    grads_mean = np.mean(params, axis=0)
    grads_stdev = np.var(params, axis=0) ** 0.5

    n = len(ordered_results)                                        # number of clients
    m = sum(val == True for val in states.values())                 # number of corrupted clients
    s = math.floor((n / 2) + 1) - m                                 # number of supporters
    z_max = norm.ppf((n - m - s) / (n - m))
    for proxy, fitres in ordered_results:
        if states[fitres.metrics["cid"]]:
            grads_mean[:] = grads_mean[:] - (z_max * grads_stdev[:])
            fitres.parameters = ndarrays_to_parameters(grads_mean)
            results[int(fitres.metrics['cid'])] = (proxy, fitres)
    return results, {}

def fang_attack(
        ordered_results:List[Tuple[ClientProxy, FitRes]], 
        states:Dict[str, bool],
        **kwargs
    ) -> List[Tuple[ClientProxy, FitRes]]:
    """
    Local Model Poisoning Attacks to Byzantine-Robust Federated Learning, Fang et al. (2020)
    Specifically designed for Krum, but they claim it works for other aggregation functions as well.
    Omniscent version.

    Input:
        ordered_results - list of tuples (client_proxy, fit_result) ordered by client id
        states - dictionary of client ids and their states (True if malicious, False otherwise)
        magnitude - magnitude of the attack
        d - number of parameters
        w_re - selected model
        old_lambda - lambda value
        threshold - threshold for lambda
        malicious_selected - whether the malicious client was selected or not
    """

    d = kwargs.get("d", 1)
    w_re = kwargs.get("w_re", None)
    old_lambda = kwargs.get("old_lambda", 0.0)
    threshold = kwargs.get("threshold", 0.0)
    malicious_selected = kwargs.get("malicious_selected", False)
    
    n = len(ordered_results)                                        # number of clients
    c = sum(val == True for val in states.values())                 # number of corrupted clients

    if old_lambda == 0:
        benign = [
            (parameters_to_ndarrays(fitres.parameters), fitres.num_examples)
            for _, fitres in ordered_results if states[fitres.metrics["cid"]] == False
        ]
        all = [
            (parameters_to_ndarrays(fitres.parameters), fitres.num_examples)
            for _, fitres in ordered_results
        ]
        # Compute the smallest distance that Krum would choose
        distances = np.array(_krum(all, c))
        print("Distances: ", distances)
        idx_benign = [int(cid) for cid in states.keys() if states[cid]==False]
        print("Benign clients idx: ", idx_benign)
        min_dist = np.min(distances[idx_benign]) / ((n - 2*c -1)*np.sqrt(d))
        print("min_dist", min_dist)

        # Compute max distance from w_re
        dist_wre = np.zeros((len(benign)))
        for i in range(len(benign)):
            dist = [benign[i][0][j] - w_re[j] for j in range(d)]
            norm_sums = 0
            for k in dist:
                norm_sums += np.linalg.norm(k)
            dist_wre[i] = norm_sums**2
        print("dist_wre", dist_wre)
        max_dist = np.max(dist_wre) / np.sqrt(d)
        print("max dist: ", max_dist)

        l = min_dist + max_dist                                         # lambda
    else:
        l = old_lambda
        if old_lambda > threshold and malicious_selected == False:
            l = old_lambda * 0.5

    # Compute sign vector s
    magnitude = []
    for i in range(len(w_re)):
        magnitude.append(np.sign(w_re[i]) * l)
    w_1 = w_re + magnitude                                               # new corrupted update
    corrupted_params = ndarrays_to_parameters(w_1)

    # Set corrupted clients' updates to w_1
    results =[
        (
            proxy, 
            FitRes(fitres.status, parameters=corrupted_params, num_examples=fitres.num_examples, metrics=fitres.metrics)
        ) if states[fitres.metrics["cid"]] else (proxy, fitres) 
        for proxy, fitres in ordered_results
    ]

    return results, {"lambda": l}

def pga_attack(
        ordered_results:List[Tuple[ClientProxy, FitRes]], 
        states:Dict[str, bool], 
        magnitude:float
    ) -> List[Tuple[ClientProxy, FitRes]]:
    """
    PGA attack, Shejwalkar et al. (2022)
    """
    pass

def _krum(results: List[Tuple[List, int]], c:int) -> NDArrays:
    """
    Get the best parameters vector according to the Krum function.
    Output: the best parameters vector.
    """
    weights = [w for w, _ in results]                                   # list of weights
    M = _compute_distances(weights)                                     # matrix of distances
    num_closest = len(weights) - c - 2                                  # number of closest points to use
    closest_indices = _get_closest_indices(M, num_closest)              # indices of closest points
    scores = [np.sum(M[i,closest_indices[i]]) for i in range(len(M))]   # scores i->j for each i
    best_index = np.argmin(scores)                                      # index of the best score
    return scores                                                       # distances

def _compute_distances(weights: NDArrays) -> NDArrays:
    """
    Compute the distance between the vectors.
    Input: weights - list of weights vectors
    Output: distances - matrix M of squared distances between the vectors
    """
    weights = np.array(weights)
    M = np.zeros((len(weights), len(weights)))
    for i in range(len(weights)):
        for j in range(len(weights)):
            d = weights[i] - weights[j]
            norm_sums = 0
            for k in d:
                norm_sums += np.linalg.norm(k)
            M[i, j] = norm_sums**2
    return M

def _get_closest_indices(M, num_closest: int) -> List[int]:
    """
    Get the indices of the closest points.
    Input: 
        M - matrix of squared distances between the vectors
        num_closest - number of closest points to get for each parameter vector
    Output:
        closest_indices - list of lists of indices of the closest points for each parameter vector 
    """
    closest_indices = []
    for i in range(len(M)):
        closest_indices.append(np.argsort(M[i])[1:num_closest+1].tolist())
    return closest_indices
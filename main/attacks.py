import math
import numpy as np
from typing import Dict, List, Tuple
from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    FitRes,
)
from flwr.server.client_proxy import ClientProxy

def gaussian_attack(
        ordered_results:List[Tuple[ClientProxy, FitRes]], 
        states:Dict[str, bool], 
        magnitude:float
    ) -> List[Tuple[ClientProxy, FitRes]]:
    results = ordered_results.copy()
    for proxy, fitres in ordered_results:
        if states[fitres.metrics["cid"]]:
            params = parameters_to_ndarrays(fitres.parameters)
            perturbate = lambda a: a + np.random.normal(loc=0, scale=magnitude, size=len(a))
            new_params = np.apply_along_axis(perturbate, 0, params)
            fitres.parameters = ndarrays_to_parameters(new_params)
            results[int(fitres.metrics['cid'])] = (proxy, fitres)
    return results

def lie_attack(
        ordered_results:List[Tuple[ClientProxy, FitRes]], 
        states:Dict[str, bool], 
        magnitude:float
    ) -> List[Tuple[ClientProxy, FitRes]]:
    results = ordered_results.copy()
    params = [parameters_to_ndarrays(fitres.parameters) for _, fitres in results]
    grads_mean = np.mean(params, axis=0)
    grads_stdev = np.var(params, axis=0) ** 0.5

    n = len(ordered_results)
    m = sum(val == True for val in states.values())
    s = math.ceil((n / 2) + 1) - m
    z_max = (n - m - s) / (n - m)
    print("z_max:", z_max)
    for proxy, fitres in ordered_results:
        if states[fitres.metrics["cid"]]:
            grads_mean[:] -= z_max * grads_stdev[:]
            fitres.parameters = ndarrays_to_parameters(grads_mean)
            results[int(fitres.metrics['cid'])] = (proxy, fitres)
    return results
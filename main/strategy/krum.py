import flwr as fl
import numpy as np
import matplotlib.pyplot as plt

from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union
from strategy.robustrategy import RobustStrategy

from flwr.common import (
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy

from utilities import flatten_params

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""

class Krum(RobustStrategy):
    """
    Configurable Krum strategy.

    Implementation of P.Blanchard et al. (2017) "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        dataset_name: str = "circles",
        strategy_name: str = "not specified",
        attack_name: str = "not specified",
        iid: bool = True,
        malicious_clients: int = 0,
        magnitude: float = 1.0,
        threshold: float = 0.005,
        warmup_rounds: int = 1,
        to_keep: int = 1,
        attack_fn: Optional[Callable],
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:
        """
        Parameters
        ----------
        fravtion_malicious : float, otional
            Fraction of malicious clients. Defaults to 0.
        """

        super().__init__(
                fraction_fit=fraction_fit, 
                fraction_evaluate = fraction_evaluate, 
                min_fit_clients = min_fit_clients, 
                min_evaluate_clients = min_evaluate_clients, 
                min_available_clients = min_available_clients,
                evaluate_fn = evaluate_fn, 
                on_fit_config_fn = on_fit_config_fn, 
                on_evaluate_config_fn = on_evaluate_config_fn,
                accept_failures = accept_failures,
                initial_parameters = initial_parameters,
                fit_metrics_aggregation_fn = fit_metrics_aggregation_fn,
                evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn,
                dataset_name = dataset_name,
                strategy_name = strategy_name,
                attack_name = attack_name,
                iid = iid,
                malicious_clients = malicious_clients,
                magnitude = magnitude,
                warmup_rounds = warmup_rounds,
                to_keep = to_keep,
                attack_fn = attack_fn,
                threshold = threshold
            )
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using median on weights."""
        results, others, clients_state = super().init_fit(server_round, results, failures)

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fitres.parameters), fitres.num_examples)
            for _, fitres in results
        ]

        #save_history_average(weights_results)
        print("client states ", clients_state)
        self.aggregated_parameters, selected_cid, _, _ = krum(weights_results, self.m[-1], self.to_keep)
        print("selected client ids ", selected_cid)
        self.malicious_selected = clients_state[selected_cid]
        print("best client: "+str(selected_cid))

        #np.save("strategy/krum_parameters_aggregated.npy", parameters_to_ndarrays(parameters_aggregated))
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return ndarrays_to_parameters(self.aggregated_parameters), metrics_aggregated

def krum(results: List[Tuple[List, int]], m: int, to_keep: int):
    """
    Get the best parameters vector according to the Krum function.
    Output: the best parameters vector.
    """
    weights = [w for w, _ in results]                             # list of weights
    M = _compute_distances(weights)                                         # matrix of distances
    num_closest = len(weights) - m - 2                                      # number of closest points to use
    closest_indices = _get_closest_indices(M, num_closest)                  # indices of closest points
    scores = [np.sum(M[i,closest_indices[i]]) for i in range(len(M))]       # scores i->j for each i
    print("scores _aggregate_weights: "+str(scores))
    best_index = np.argmin(scores)                                          # index of the best score
    best_indices = np.argsort(scores)[::-1][len(scores)-to_keep:]           # indices of best scores (multikrum)
    return weights[best_index], best_index, best_indices, scores            # best weights vector

def _compute_distances(weights: NDArrays) -> NDArrays:
    """
    Compute the distance between the vectors.
    Input: weights - list of weights vectors
    Output: distances - matrix M of squared distances between the vectors
    """
    #weights = np.array(weights)
    w = np.array([flatten_params(p) for p in weights])
    M = np.zeros((len(weights), len(weights)))
    for i in range(len(w)):
        for j in range(len(w)):
            d = w[i] - w[j]
            norm = np.linalg.norm(d)
            M[i, j] = norm**2
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
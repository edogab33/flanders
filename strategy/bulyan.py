from functools import reduce
from random import sample
import flwr as fl
import numpy as np

from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union
from strategy.robustrategy import RobustStrategy
from strategy.krum import krum
from strategy.fedmedian import compute_median_vect

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
from flwr.server.strategy.aggregate import aggregate

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""

class Bulyan(RobustStrategy):
    """
    Configurable Bulyan strategy implementation.

    From: The Hidden Vulnerability of Distributed Learning in Byzantium, El Mhamdi et al.
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
        window: int = 0,
        sampling: str = None,
        configs: Optional[Dict[str, str]] = None,
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
            (parameters_to_ndarrays(fit_res.parameters), fit_res.metrics["cid"])
            for _, fit_res in results
        ]

        theta = self.sample_size[-1] - 2*self.m[-1]
        if theta <= 0:
            theta = 1
        print("theta: ", theta)
        beta = theta - 2*self.m[-1]
        if beta <= 0:
            beta = 1
        print("beta: ", beta)

        S = {}                                                                      # S must be Dict[int, Tuple[NDArrays, int]]
        tracker = np.arange(len(weights_results))                                   # List of idx to keep track of the order of clients
        for _ in range(theta):
            _, idx, _, _ = krum(weights_results, self.m[-1], self.to_keep)
            S[tracker[idx]] = weights_results[idx]                                  # weights_results is ordered according to "cid"

            # remove idx from tracker and weights_results
            tracker = np.delete(tracker, idx)
            weights_results.pop(idx)

        # Compute median parameter vector across S
        median_vect = compute_median_vect(S.values())

        # Take the beta closest params to the median
        distances = {}
        for i in S.keys():
            dist = [
                np.abs(S[i][0][j] - median_vect[j]) for j in range(len(self.aggregated_parameters))
            ]
            norm_sums = 0
            for k in dist:
                norm_sums += np.linalg.norm(k)
            distances[i] = norm_sums

        closest_idx = sorted(distances, key=distances.get)[:beta]
        M = [S[i][0] for i in closest_idx]
        print("selected ", closest_idx)
        print("states ", clients_state)

        for idx in closest_idx:
            if clients_state[idx]:
                self.malicious_selected = True
                break
            else:
                self.malicious_selected = False

        # Apply FevAvg on M
        parameters_aggregated: NDArrays = [
            reduce(np.add, layers) / beta
            for layers in zip(*M)
        ]

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return ndarrays_to_parameters(parameters_aggregated), metrics_aggregated
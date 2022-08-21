import flwr as fl
import numpy as np
import json
import pandas as pd


from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

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
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.strategy.strategy import Strategy

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""


class Krum(fl.server.strategy.FedAvg):
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
        fraction_malicious: float = 0.0,
        magnitude: float = 1.0,
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
                evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
            )

        if (
            self.min_fit_clients > self.min_available_clients
            or self.min_evaluate_clients > self.min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_malicious = fraction_malicious
        self.magnitude = magnitude
        self.aggr_losses = np.array([])
        self.f = 0
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)

        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        self.f = int(sample_size * self.fraction_malicious)

        print("sample size: "+str(sample_size))
        print("num f: "+str(self.f))

        fit_ins_array = [
            FitIns(parameters, dict(config, **{"malicious": True, "magnitude": self.magnitude}) if idx < self.f else dict(config, **{"malicious": False}))
            for idx,_ in enumerate(clients)]

        return [(client, fit_ins_array[idx]) for idx,client in enumerate(clients)]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using median on weights."""

        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # For test_strategy
        #weights_results = [
        #    (fit_res.parameters, fit_res.num_examples)
        #    for _, fit_res in results
        #]

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        #with open("weights_results.pkl", "w") as fp:
        #    json.dump(pd.Series(weights_results).to_json(orient='split'), fp)

        parameters_aggregated = ndarrays_to_parameters(self._aggregate_weights(weights_results))
        np.save("strategy/krum_parameters_aggregated.npy", parameters_to_ndarrays(parameters_aggregated))
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated


    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        self.aggr_losses = np.append(loss_aggregated, self.aggr_losses)
        np.save("/Users/eddie/Documents/Università/ComputerScience/Thesis/flwr-pytorch/results/aggregated_losses.npy", self.aggr_losses)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated

    def _aggregate_weights(self, results: List[Tuple[List, int]]) -> NDArrays:
        """
        Get the best parameters vector according to the Krum function.

        Output: the best parameters vector.
        """
        weights = [weights for weights, _ in results]                   # list of weights
        M = self._compute_distances(weights)                            # matrix of distances
        num_closest = len(weights) - self.f - 2                         # number of closest points to use
        closest_indices = self._get_closest_indices(M, num_closest)     # indices of closest points
        scores = [np.sum(d) for d in closest_indices]                   # scores i->j for each i
        best_index = np.argmin(scores)                                  # index of the best score
        return weights[best_index]                                      # best weights vector

    def _compute_distances(self, weights: NDArrays) -> NDArrays:
        """
        Compute the distance between the vectors.

        Input: weights - list of weights vectors
        Output: distances - matrix M of squared distances between the vectors
        """
        M = np.zeros((len(weights), len(weights)))
        for i in range(len(weights)):
            for j in range(len(weights)):
                d = weights[i] - weights[j]
                norm_sums = 0
                for k in d:
                    norm_sums += np.linalg.norm(k, ord=1)**2
                M[i, j] = norm_sums
        return M

    def _get_closest_indices(self, M, num_closest: int) -> List[int]:
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
            closest_indices.append(np.argpartition(M[i], num_closest)[:num_closest])
        return closest_indices
import flwr as fl
import numpy as np
from functools import reduce

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

from flwr.server.strategy.aggregate import aggregate

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""

class MultiKrum(RobustStrategy):
    """
    Configurable Multi-Krum strategy.

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
                fraction_malicious = fraction_malicious,
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

        results, others, clients_state = super().init_fit(server_round, results, failures)        # Convert results
        
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        # Take the m best parameters vectors and average them
        best_indices = self._get_best_parameters(weights_results)
        weights_results = [weights_results[idx] for idx in best_indices]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))
        np.save("strategy/multikrum_parameters_aggregated.npy", parameters_to_ndarrays(parameters_aggregated))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def _get_best_parameters(self, results: List[Tuple[int, float]]) -> NDArrays:
        """
        Get the best parameters vectors according to the Multi-Krum function.

        Output: the m best parameters vectors.
        """
        weights = [weights for weights, _ in results]                                   # list of weights
        M = self._compute_distances(weights)                                            # matrix of distances
        num_closest = len(weights) - self.m[-1] - 2                                     # number of closest points to use
        closest_indices = self._get_closest_indices(M, num_closest)                     # indices of closest points
        scores = [np.sum(M[i,closest_indices[i]]) for i in range(len(M))]               # scores i->j for each i
        best_indices = np.argsort(scores)[::-1][len(scores)-self.to_keep:]  # indices of best scores
        return best_indices

    def _compute_distances(self, weights: NDArrays) -> NDArrays:
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
            closest_indices.append(np.argsort(M[i])[1:num_closest+1].tolist())
        return closest_indices

    def _aggregate(self, results: List[Tuple[NDArrays, int]]) -> NDArrays:
        """Compute weighted average."""
        # Calculate the total number of examples used during training
        num_examples_total = sum([num_examples for _, num_examples in results])

        # Create a list of weights, each multiplied by the related number of examples
        weighted_weights = [
            [layer * num_examples for layer in weights] for weights, num_examples in results
        ]

        # Compute average weights of each layer
        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]
        return weights_prime
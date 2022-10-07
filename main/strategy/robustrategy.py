from http import server
import flwr as fl
import numpy as np
import matplotlib.pyplot as plt

from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union
from strategy.utilities import (
    evaluate_aggregated, 
    save_params, 
    flatten_params
)
from neural_networks.dataset_utils import get_circles

from flwr.common import (
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
from flwr.server.strategy.aggregate import aggregate

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""

class RobustStrategy(fl.server.strategy.FedAvg):
    """
    Configurable robust strategy. Used as a superclass for other robust strategies.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        magnitude: float = 1.0,
        threshold: float = 0.005,
        warmup_rounds: int = 1,
        to_keep: int = 1,
        attack_fn: Optional[Callable],
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        dataset_name: str = "not specified",
        strategy_name: str = "not specified",
        attack_name: str = "not specified",
        iid: bool = True,
        malicious_clients: int = 0,
        window: int = 0,
        sampling: int = 0,
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
            )

        if (
            self.min_fit_clients > self.min_available_clients
            or self.min_evaluate_clients > self.min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.malicious_clients = malicious_clients
        self.magnitude = magnitude
        self.aggr_losses = np.array([])
        self.m = [0]                                                # number of malicious clients (updates each round)
        self.sample_size = [1]                                      # number of clients available (updates each round)
        self.cm = [[0,0],[0,0]]                                     # confusion matrix (updates each round)
        self.attack_fn = attack_fn                                  # attack function
        self.aggregated_parameters = []                             # global model (updates each round)
        self.malicious_selected = False                             # selected malicious parameters in this round? (updates each round)
        self.old_lambda = 0.0                                       # lambda from previous round (updates each round)
        if attack_name == "minmax":
            self.old_lambda = 5.0
        self.warmup_rounds = warmup_rounds                          # number of warmup rounds
        self.to_keep = to_keep                                      # number of cliernts to aggregate
        self.threshold = threshold                                  # threshold for fang and minmax attacks
        self.dataset_name = dataset_name.lower()                    # dataset name used by clients (circles, mnist, cifar10, etc.)
        self.root_dataset = None                                    # root dataset used by the server (circles, mnist, cifar10, etc.)
        self.strategy_name = strategy_name.lower()                  # strategy name (fedavg, krum, etc.)
        self.attack_name = attack_name.lower()                      # attack name (gaussian, lie, etc.)
        self.iid = iid                                              # iid or non-iid dataset
        self.window = window                                        # window size (num of timesteps loaded and window size for MAR)
        self.sampling = sampling                                    # number of params to sample
        self.params_indexes = []                                    # indexes of sampled parameters after round 1

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
        self.sample_size.append(sample_size)
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        self.m.append(int(self.malicious_clients))

        print("sample size: "+str(sample_size))
        print("num m: "+str(self.m[-1]))

        fit_ins_array = [
            FitIns(parameters, dict(config, **{"malicious": True, "magnitude": self.magnitude}) if idx < self.m[-1] else dict(config, **{"malicious": False}))
            for idx,_ in enumerate(clients)
        ]

        return [(client, fit_ins_array[idx]) for idx, client in enumerate(clients)]

    def init_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Initialize the robust aggregation and apply the attack function."""

        print("FAILURES: ", failures)
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

        clients_state = {}      # dictionary of clients representing wether they are malicious or not

        # Save parameters of each client as time series
        ordered_results = [0 for _ in range(len(results))]
        cids = np.array([])
        for proxy, fitres in results:
            cids = np.append(cids, int(fitres.metrics["cid"]))
            clients_state[fitres.metrics['cid']] = fitres.metrics['malicious']
            params = flatten_params(parameters_to_ndarrays(fitres.parameters))
            if self.sampling > 0:
                if len(self.params_indexes) == 0:
                    # Sample a random subset of parameters
                    self.params_indexes = np.random.randint(0, len(params), size=self.sampling)
                params = params[self.params_indexes]
            save_params(params, fitres.metrics['cid'])
            # Re-arrange results in the same order as clients' cids impose
            ordered_results[int(fitres.metrics['cid'])] = (proxy, fitres)

        if self.aggregated_parameters == []:
            # Initialize aggregated_parameters if it is the first round
            for key, val in clients_state.items():
                if val == False:
                    self.aggregated_parameters = parameters_to_ndarrays(ordered_results[int(key)][1].parameters)
                    break

        if self.root_dataset == None:    
            # Load the root dataset
            if self.dataset_name == "circles":
                self.root_dataset = get_circles(32, n_samples=10000, is_train=True)
            elif self.dataset_name == "mnist":
                #TODO: Implement
                pass
            elif self.dataset_name == "cifar":
                #TODO: Implement
                pass

        if server_round > self.warmup_rounds:
            results, others = self.attack_fn(
                ordered_results, clients_state, magnitude=self.magnitude,
                w_re=self.aggregated_parameters, malicious_selected=self.malicious_selected,
                threshold=self.threshold, d=len(self.aggregated_parameters), old_lambda=self.old_lambda,
                dataset_name=self.dataset_name, agr_function=self.strategy_name, to_keep = self.to_keep,
                malicious_num=self.m[-1]
            )
            self.old_lambda = others.get('lambda', 0.0)

            # Update saved parameters time series after the attack
            for proxy, fitres in results:
                if self.sampling > 0:
                    params = flatten_params(parameters_to_ndarrays(fitres.parameters))[self.params_indexes]
                else:
                    params = flatten_params(parameters_to_ndarrays(fitres.parameters))
                save_params(params, fitres.metrics['cid'], remove_last=True)
        else:
            results = ordered_results
            others = {}
        
        return results, others, clients_state

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None

        config = {"strategy": self.strategy_name, "fraction_mal": self.malicious_clients, "magnitude": self.magnitude, 
            "frac_fit": self.fraction_fit, "frac_eval": self.fraction_evaluate, "min_fit_clients": self.min_fit_clients,
            "min_eval_clients": self.min_evaluate_clients, "min_available_clients": self.min_available_clients,
            "num_clients": self.sample_size[-1], "num_malicious": self.m[-1], "attack": self.attack_name, "iid": self.iid,
            "dataset_name": self.dataset_name, "confusion_matrix": self.cm, "warmup_rounds": self.warmup_rounds}

        eval_res = evaluate_aggregated(self.evaluate_fn, server_round, parameters, config)
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def _aggregate_weights(self, results: List[Tuple[List, int]]) -> NDArrays:
        """
        Get the best parameters vector according to the Krum function.

        Output: the best parameters vector.
        """
        weights = [np.array(w) for w, _ in results]             # list of weights
        M = self._compute_distances(weights)                                # matrix of distances
        num_closest = len(weights) - self.m[-1] - 2                         # number of closest points to use
        closest_indices = self._get_closest_indices(M, num_closest)         # indices of closest points
        scores = [np.sum(M[i,closest_indices[i]]) for i in range(len(M))]   # scores i->j for each i
        print("scores _aggregate_weights: "+str(scores))
        best_index = np.argmin(scores)                                      # index of the best score
        return weights[best_index], best_index                              # best weights vector

    def _compute_distances(self, weights: NDArrays) -> NDArrays:
        """
        Compute the distance between the vectors.

        Input: weights - list of weights vectorsa
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
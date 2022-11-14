from http import server
from locale import normalize
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
from flwr.server.strategy.aggregate import aggregate

from clients import (
    CifarClient, 
    HouseClient, 
    IncomeClient, 
    ToyClient, 
    set_params, 
    get_params, 
    MnistClient, 
    set_sklearn_model_params, 
    get_sklearn_model_params
)
from neural_networks.dataset_utils import (
    get_mnist, 
    do_fl_partitioning, 
    get_dataloader, 
    get_circles, 
    get_cifar_10, 
    get_partitioned_house, 
    get_partitioned_income
)

from scipy import spatial

from strategy.utilities import flatten_params

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""

class FLTrust(RobustStrategy):
    """
    Configurable FLTrust strategy.

    Implementation of FLTrust: 
    "Byzantine-robust Federated Learning via Trust Bootstrapping", Cao et al. (2020)
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
        
        self.previous_params = []

        if dataset_name == "circles":
            client_func = ToyClient
        elif dataset_name == "mnist":
            client_func = MnistClient
        elif dataset_name == "cifar":
            client_func = CifarClient
        elif dataset_name == "income":
            client_func = IncomeClient
        elif dataset_name == "house":
            client_func = HouseClient

        if dataset_name == "cifar":
            # Cifar-10 dataset
            train_path, testset = get_cifar_10()
            fed_dir = do_fl_partitioning(
                train_path, pool_size=1, alpha=1000, num_classes=10, val_ratio=0.1
            )
        elif dataset_name == "income":
            # Income dataset
            X_train, X_test, y_train, y_test = get_partitioned_income("datasets/adult.csv", 1)
        elif dataset_name == "house":
            # House prediction dataset
            X_train, X_test, y_train, y_test = get_partitioned_house("datasets/houses_preprocessed.csv", 1)

        # create a single client instance
        if dataset_name == "cifar":
            self.client = CifarClient(0, fed_dir)
        elif dataset_name == "income":
            self.client = IncomeClient(0, X_train[0], y_train[0], X_test[0], y_test[0])
        elif dataset_name == "house":
            self.client = HouseClient(0, X_train[0], y_train[0], X_test[0], y_test[0])
        else:
            self.client = client_func(0, 1)
    
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

        if server_round == 1:
            # Take a parameter vector from one benign client
            for key,val in clients_state.items():
                if val == False:
                    self.previous_params = weights_results[int(key)][0]
                    break

        # Take one learning step
        g0, _, _ = self.client.fit(self.previous_params, {"epochs": 1, "malicious": False, "batch_size":64})
        self.previous_params = g0

        print("client state: ", clients_state)
        
        # Compute cosine similarities between g0 and all other clients
        c_i = []
        flattened = [flatten_params(x[0]) for x in weights_results]

        g0_flattened = flatten_params(g0)
        for i in range(len(flattened)):
            c_i.append(spatial.distance.cosine(flattened[i], g0_flattened))

        # Apply ReLU to obtain cosine similarities
        trust_scores = [x if x > 0 else 0 for x in c_i]

        # Normilize parameters
        normalized_params = weights_results.copy()
        g0_norm = 0
        for layer in g0:
            g0_norm += np.linalg.norm(layer)
        print("g0_norm", g0_norm)
        for i in range(len(weights_results)):
            gi_norm = 0
            for layer in weights_results[i][0]:
                gi_norm += np.linalg.norm(layer)
            print("gi_norm: ", gi_norm)
            normalized_params[i] = ([(g0_norm / gi_norm) * layer for layer in weights_results[i][0]], trust_scores[i])
        print("trust scores", trust_scores)

        # Aggregate parameters weighted by trust scores
        update = aggregate(normalized_params)
        for layer in range(len(update)):
            self.aggregated_parameters[layer] = self.aggregated_parameters[layer] + update[layer]

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return ndarrays_to_parameters(self.aggregated_parameters), metrics_aggregated
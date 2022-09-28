import flwr as fl
import numpy as np
import pandas as pd

from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import (
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

import strategy.mscred.evaluate as eval
import strategy.mscred.matrix_generator as mg

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""

class LocalFlanders(fl.server.strategy.FedAvg):
    """
    Aggregation function based on MSCRED anomaly detection.
    This is the Global Approach, where parameters trained by 
    each client are analyzed to detect anomalies within the client itself.
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
        """Apply MSCRED to exclude malicious clients from the average."""
        results, others, clients_state = super().init_fit(server_round, results, failures)

        weights_results = {
            proxy.cid: np.asarray(parameters_to_ndarrays(fit_res.parameters))
            for proxy, fit_res in results
        }
        
        params = np.asarray([])
        for cid in weights_results:
            weights = weights_results[cid][0].reshape((*weights_results[cid][0].shape[:-2], -1))[:200]
            
            # Instead of making time series of TxN make them CxN where C is the number of clients
            params = np.vstack((params, weights)) if params.size else weights
        np.save("params_ts.npy", params)
        df = pd.DataFrame(params.T)
        df.to_csv("/Users/eddie/Documents/Università/ComputerScience/Thesis/flwr-pytorch/main/params_ts.csv", index=False, header=False)

        # For each client, make signature test matrices
        mg.generate_train_test_data(test_end=params.shape[0], step_max=5)

        # Load MSCRED trained model and generate reconstructed matrices
        mg.generate_reconstructed_matrices(test_end_id=params.shape[0], sensor_n=params.shape[1], step_max=5)

        # Compute anomaly scores
        anomaly_scores = np.array(eval.evaluate(test_end_point=params.shape[0], threshold=self.threshold))
    
        results = [sorted(np.argsort(anomaly_scores)[:self.to_keep])]

        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        return parameters_aggregated, metrics_aggregated
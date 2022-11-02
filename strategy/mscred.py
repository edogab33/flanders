import flwr as fl
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union
from strategy.robustrategy import RobustStrategy
from strategy.utilities import evaluate_aggregated

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
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

import strategy.mscred_utils.evaluate as eval
import strategy.mscred_utils.matrix_generator as mg

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""


class Mscred(RobustStrategy):
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
        window: int = 0,
        sampling: str = None,
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
        HPATH = "strategy/mscred_utils/histories/"
        if not os.path.exists(HPATH):
            os.makedirs(HPATH)
        weights_results = {
            proxy.cid: np.asarray(parameters_to_ndarrays(fit_res.parameters))
            for proxy, fit_res in results
        }
        
        params = np.asarray([])
        for cid in weights_results:
            flattened_params = np.concatenate([w.flatten() for w in weights_results[cid]])
            print(np.mean(flattened_params))
            params = np.append(params, np.mean(flattened_params))

        #np.save("params_ts.npy", params)

        # check that strategy/histoies directory exists and load history if it does
        history = np.load(HPATH+"history.npy") if os.path.exists(HPATH+"history.npy") else np.array([])
        history = np.vstack((history, params)) if history.size else params
        np.save(HPATH+"history.npy", history)
        
        if server_round >= self.warmup_rounds:
            df = pd.DataFrame(history.T)
            df.to_csv(HPATH+"history.csv", index=False, header=False)

            # For each client, make signature test matrices
            mg.generate_train_test_data(test_start=server_round-10, test_end=server_round, step_max=5, win_size=[1], params_time_series=HPATH+"history.csv",
                gap_time=1)

            # Load MSCRED trained model and generate reconstructed matrices
            mg.generate_reconstructed_matrices(test_start_id=server_round-10, test_end_id=server_round, sensor_n=history.shape[1], step_max=5, scale_n=3,
                model_path="strategy/mscred_utils/model_ckpt/8/", restore_idx=13)

            # Compute anomaly scores
            anomaly_scores = np.array(eval.evaluate(threshold=self.threshold, test_matrix_id=server_round-1))
            print(anomaly_scores)
            # Keep only the 'to_keep' clients with lower socres
            print(sorted(np.argsort(anomaly_scores)[:self.to_keep]))
            results = np.array(results)[sorted(np.argsort(anomaly_scores)[:self.to_keep])].tolist()

        # TODO: save history without malicious clients (?)
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        return parameters_aggregated, metrics_aggregated
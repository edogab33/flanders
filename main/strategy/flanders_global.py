import flwr as fl
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

from typing import Callable, Dict, List, Optional, Tuple, Union
from strategy.robustrategy import RobustStrategy
from strategy.utilities import (
    save_params, 
    load_all_time_series, 
    load_time_series, 
    update_confusion_matrix, 
    flatten_params
)

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

class GlobalFlanders(RobustStrategy):
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
                threshold = threshold,
                window = window
            )

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Apply MAR forecasting to exclude malicious clients from the average.
        """
        results, others, clients_state = super().init_fit(server_round, results, failures)

        if server_round > self.warmup_rounds:
            M = load_all_time_series(dir="clients_params", window=self.window)
            M = np.transpose(M, (0, 2, 1))
            M_hat = M[:,:,-1].copy()
            pred_step = 1
            Mr = self.mar(M[:,:,:-1], pred_step, window=0)
            select_matrix_error = np.square(np.subtract(M_hat, Mr[:,:,0]))
            num_broken = len(select_matrix_error[select_matrix_error > self.threshold])
            print("Overall anomaly score: ", num_broken)

            anomaly_scores = []
            #compute anomaly score for each client
            for client in select_matrix_error:
                anomaly_scores.append(np.sum(client))
            print("Anomaly scores: ", anomaly_scores)
            good_clients_idx = sorted(np.argsort(anomaly_scores)[:self.to_keep])
            malicious_clients_idx = sorted(np.argsort(anomaly_scores)[self.to_keep:])
            results = np.array(results)[good_clients_idx].tolist()

            print("Clients kept: ")
            print(good_clients_idx)
            print("Clients: ")
            print(clients_state)

            self.cm = update_confusion_matrix(self.cm, clients_state, good_clients_idx, malicious_clients_idx)

            #fig, ax = plt.subplots(1,3, figsize=(10,5))
            #ax[0].matshow(M_hat)
            #ax[1].matshow(Mr[:,:,0])
            #ax[2].matshow(select_matrix_error)
            #plt.show()

            # Aplly FedAvg for the remaining clients
            parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

            # For clients detected as malicious, set their parameters to be the averaged ones in their files
            # otherwise the forecasting in next round won't be reliable
            for idx in malicious_clients_idx:
                save_params(flatten_params(parameters_to_ndarrays(parameters_aggregated)), idx, remove_last=True)
        else:
            parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        return parameters_aggregated, metrics_aggregated

    def mar(self, X, pred_step, maxiter = 100, window = 0):
        m, n, T = X.shape
        if window > 0:
            T = window
        B = np.random.randn(n, n)
        for it in range(maxiter):
            temp0 = B.T @ B
            temp1 = np.zeros((m, m))
            temp2 = np.zeros((m, m))
            for t in range(1, T):
                temp1 += X[:, :, t] @ B @ X[:, :, t - 1].T
                temp2 += X[:, :, t - 1] @ temp0 @ X[:, :, t - 1].T
            A = temp1 @ np.linalg.inv(temp2)
            temp0 = A.T @ A
            temp1 = np.zeros((n, n))
            temp2 = np.zeros((n, n))
            for t in range(1, T):
                temp1 += X[:, :, t].T @ A @ X[:, :, t - 1]
                temp2 += X[:, :, t - 1].T @ temp0 @ X[:, :, t - 1]
            B = temp1 @ np.linalg.inv(temp2)
        tensor = np.append(X, np.zeros((m, n, pred_step)), axis = 2)
        for s in range(pred_step):
            tensor[:, :, T + s] = A @ tensor[:, :, T + s - 1] @ B.T
        return tensor[:, :, - pred_step :]

    def aggregate_fit_mscred(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Apply MSCRED to exclude malicious clients from the average."""
        print("round: "+str(server_round))
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Save params for initial_parameters
        #weights_results = [
        #    (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
        #    for _, fit_res in results
        #]
        #parameters_aggregated = aggregate(weights_results)
        #np.save("/Users/eddie/Documents/Università/ComputerScience/Thesis/flwr-pytorch/main/strategy/histories/aggregated_params.npy", parameters_aggregated)

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
        history = np.load("strategy/histories/history.npy") if os.path.exists("strategy/histories/history.npy") else np.array([])
        history = np.vstack((history, params)) if history.size else params
        np.save("strategy/histories/history.npy", history)
        
        if server_round >= self.warmup_rounds:
            df = pd.DataFrame(history.T)
            df.to_csv("strategy/histories/history.csv", index=False, header=False)

            # For each client, make signature test matrices
            mg.generate_train_test_data(test_start=server_round-10, test_end=server_round, step_max=5, win_size=[10,30,60], params_time_series="strategy/histories/history.csv",
                gap_time=1)

            # Load MSCRED trained model and generate reconstructed matrices
            mg.generate_reconstructed_matrices(test_start_id=server_round-10, test_end_id=server_round, sensor_n=history.shape[1], step_max=5, scale_n=9,
                model_path="strategy/mscred/model_ckpt/8/", restore_idx=18)

            # Compute anomaly scores
            anomaly_scores = np.array(eval.evaluate(threshold=self.threshold, test_matrix_id=server_round-1))
            print(anomaly_scores)
            # Keep only the 'to_keep' clients with lower socres
            print(sorted(np.argsort(anomaly_scores)[:self.to_keep]))
            results = np.array(results)[sorted(np.argsort(anomaly_scores)[:self.to_keep])].tolist()

        # TODO: save history without malicious clients (?)
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        return parameters_aggregated, metrics_aggregated
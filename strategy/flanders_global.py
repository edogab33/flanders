import flwr as fl
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from typing import Callable, Dict, List, Optional, Tuple, Union
from strategy.robustrategy import RobustStrategy
from strategy.utilities import (
    save_params, 
    save_predicted_params,
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

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""

class GlobalFlanders(RobustStrategy):
    """
    Aggregation function based on MAR.
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
                threshold = threshold,
                window = window,
                sampling=sampling
            )

        self.alpha = configs.get("alpha", 0.1)
        self.beta = configs.get("beta", 0.1)

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

        if server_round > 1:
            win = self.window
            if server_round < self.window:
                win = server_round
            M = load_all_time_series(dir="strategy/clients_params", window=win)
            M = np.transpose(M, (0, 2, 1))  # (clients, params, time)

            M_hat = M[:,:,-1].copy()
            pred_step = 1
            
            Mr = mar(M[:,:,:-1], pred_step, maxiter=100, alpha=self.alpha, beta=self.beta)

            #for c in range(len(Mr)):
            #    params = flatten_params(Mr[c])
            #    if self.sampling > 0:
            #        params = params[self.params_indexes]
            #    save_predicted_params(params, c)

            delta = np.subtract(M_hat, Mr[:,:,0])
            anomaly_scores = np.sum(delta**2,axis=-1)**(1./2)
            print("Anomaly scores: ", anomaly_scores)

            good_clients_idx = sorted(np.argsort(anomaly_scores)[:self.to_keep])
            malicious_clients_idx = sorted(np.argsort(anomaly_scores)[self.to_keep:])
            results = np.array(results)[good_clients_idx].tolist()

            for idx in good_clients_idx:
                if clients_state[idx]:
                    self.malicious_selected = True
                    print("[!] Malicious client(s) selected")
                    break
                else:
                    self.malicious_selected = False

            print("Kept clients: ")
            print(good_clients_idx)
            print("Clients: ")
            print(clients_state)

            self.cm = update_confusion_matrix(self.cm, clients_state, good_clients_idx, malicious_clients_idx)

            _, ax = plt.subplots(1,3, figsize=(10,5))
            ax[0].matshow(M_hat)
            ax[1].matshow(Mr[:,:,0])
            ax[2].matshow(delta)
            plt.savefig("results/"+str(server_round)+"_matrices.png")

            # Aplly FedAvg for the remaining clients
            parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

            # For clients detected as malicious, set their parameters to be the averaged ones in their files
            # otherwise the forecasting in next round won't be reliable
            if self.warmup_rounds > server_round:
                for idx in malicious_clients_idx:
                    if self.sampling > 0:
                        new_params = flatten_params(parameters_to_ndarrays(parameters_aggregated))[self.params_indexes]
                    else:
                        new_params = flatten_params(parameters_to_ndarrays(parameters_aggregated))
                    save_params(new_params, idx, remove_last=True, rrl=False)
        else:
            parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        return parameters_aggregated, metrics_aggregated

def mar(X, pred_step, alpha=1, beta=1, maxiter=100, window=0):
   
    m, n, T = X.shape
    start = 0
    if window > 0:
        start = T - window

    A = np.random.randn(m, m)
    B = np.random.randn(n, n)
    X_norm = (X-np.min(X))/np.max(X)
   
    for it in range(maxiter):
       
        temp0 = B.T @ B
        temp1 = np.zeros((m, m))
        temp2 = np.zeros((m, m))
        identity_m = np.identity(m)
       
        for t in tqdm(range(start, T)):
            temp1 += X_norm[:, :, t] @ B @ X_norm[:, :, t - 1].T
            temp2 += X_norm[:, :, t - 1] @ temp0 @ X_norm[:, :, t - 1].T

        temp2 += (alpha * identity_m)
        A = temp1 @ np.linalg.inv(temp2)
       
        temp0 = A.T @ A
        temp1 = np.zeros((n, n))
        temp2 = np.zeros((n, n))
        identity_n = np.identity(n)
       
        for t in tqdm(range(start, T)):
            temp1 += X_norm[:, :, t].T @ A @ X_norm[:, :, t - 1]
            temp2 += X_norm[:, :, t - 1].T @ temp0 @ X_norm[:, :, t - 1]

        temp2 += (beta * identity_n)
        B = temp1 @ np.linalg.inv(temp2)
   
    tensor = np.append(X, np.zeros((m, n, pred_step)), axis = 2)
    for s in range(pred_step):
        tensor[:, :, T + s] = A @ tensor[:, :, T + s - 1] @ B.T
    return tensor[:, :, - pred_step :]
    
def cap_values(matrix):
    """
    Cap values of matrices in order to avoid
    hitting the limit of the floating point precision
    and to avoid singular matrices
    """
    #matrix = np.nan_to_num(matrix, nan=np.finfo(np.float64).max)
    matrix[matrix < np.finfo(np.float64).tiny] = np.finfo(np.float64).tiny
    return matrix

#M = load_all_time_series(dir="strategy/clients_params", window=5)
#M = np.transpose(M, (0, 2, 1))
#M_hat = M[:,:,-1].copy()
#pred_step = 1
#Mr = mar(M[:,:,:-1], pred_step, window=4, maxiter=60)
#delta = np.subtract(M_hat, Mr[:,:,0])
#anomaly_scores = np.sum(np.abs(delta)**2,axis=-1)**(1./2)
#print("Anomaly scores: ", anomaly_scores)
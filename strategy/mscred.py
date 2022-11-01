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
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.strategy.strategy import Strategy

import strategy.mscred.evaluate as eval
import strategy.mscred.matrix_generator as mg
import strategy.utilities as utils

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
        
        self.m.append(0)
        if server_round >= self.warmup_rounds:
            self.m[-1] = int(sample_size * self.fraction_malicious)

        print("sample size: "+str(sample_size))
        print("num m: "+str(self.m[-1]))

        fit_ins_array = [
            FitIns(parameters, dict(config, **{"malicious": True, "magnitude": self.magnitude}) 
            if idx < self.m[-1] and server_round >= self.warmup_rounds else dict(config, **{"malicious": False}))
            for idx,_ in enumerate(clients)]

        return [(client, fit_ins_array[idx]) for idx,client in enumerate(clients)]

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

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        config = {"strategy": "Flanders", "fraction_mal": self.fraction_malicious, "magnitude": self.magnitude, 
            "frac_fit": self.fraction_fit, "frac_eval": self.fraction_evaluate, "min_fit_clients": self.min_fit_clients,
            "min_eval_clients": self.min_evaluate_clients, "min_available_clients": self.min_available_clients,
            "num_clients": self.sample_size, "num_malicious": self.m, "confusion_matrix": self.cm}
        eval_res = evaluate_aggregated(self.evaluate_fn, server_round, parameters, config)
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

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

        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)

        self.aggr_losses = np.append(loss_aggregated, self.aggr_losses)

        return loss_aggregated, metrics_aggregated

    def flatten_params(self, params):
        params_flattened = []
        for i in range(len(params)):
            params_flattened.append([])
            for j in range(len(params[i])):
                if j == 0 or j == 2:
                    p = np.hstack(params[i][j])
                    for k in range(len(p)):
                        params_flattened[i].append(p[k])
        return params_flattened

    def load_all_time_series(self, dir=""):
        """
        Load all time series in order to have a tensor of shape (T,m,n)
        where:
        - T := time;
        - m := number of clients;
        - n := number of parameters
        """
        files = os.listdir(dir)
        files.sort()
        data = []
        for file in files:
            data.append(np.load(os.path.join(dir, file), allow_pickle=True))
        return np.array(data)

    def load_time_series(self, dir="", cid=0):
        """
        Load time series of client cid in order to have a matrix of shape (T,n)
        where:
        - T := time;
        - n := number of parameters
        """
        files = os.listdir(dir)
        files.sort()
        data = []
        for file in files:
            if file == f"{cid}.npy":
                data.append(np.load(os.path.join(dir, file), allow_pickle=True))
        return np.array(data)    

    def save_params(self, parameters, cid, remove_last=False):
        new_params = parameters
        # Save parameters in client_params/cid_params
        path = f"clients_params/{cid}_params.npy"
        if os.path.exists(path):
            # load old parameters
            old_params = np.load(path, allow_pickle=True)
            if remove_last:
                old_params = old_params[:-1]
            # add new parameters
            new_params = np.vstack((old_params, new_params))
        # save parameters
        np.save(path, new_params)
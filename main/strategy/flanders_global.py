import flwr as fl
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union
from strategy.utilities import (
    evaluate_aggregated, 
    save_params, 
    load_all_time_series, 
    load_time_series, 
    update_confusion_matrix, 
    flatten_params
)

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


class GlobalFlanders(fl.server.strategy.FedAvg):
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
                evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
            )

        if (
            self.min_fit_clients > self.min_available_clients
            or self.min_evaluate_clients > self.min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_malicious = fraction_malicious
        self.magnitude = magnitude
        self.threshold = threshold
        self.warmup_rounds = warmup_rounds                          # number of rounds needed to prepare the time series
        self.to_keep = to_keep                                      # number of clients to keep in the aggregation
        self.aggr_losses = np.array([])
        self.m = []                                                 # number of malicious clients (updates each round)
        self.sample_size = []                                       # number of clients available (updates each round)
        self.cm = [[0,0],[0,0]]                                     # confusion matrix (updates each round)
        self.attack_fn = attack_fn                                  # attack function
        self.aggregated_parameters = []                             # global model (updates each round)
        self.malicious_selected = False                             # selected malicious parameters? (updates each round)
    
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
        """
        Apply MAR forecasting to exclude malicious clients from the average.
        """
        clients_state = {}      # dictionary of clients' representing wether they are malicious or not

        # Save parameters of each client as a time series
        ordered_results = [0 for _ in range(len(results))]
        cids = np.array([])
        for proxy, fitres in results:
            cids = np.append(cids, int(fitres.metrics["cid"]))
            clients_state[fitres.metrics['cid']] = fitres.metrics['malicious']
            params = flatten_params(parameters_to_ndarrays(fitres.parameters))
            save_params(params, fitres.metrics['cid'])
            # Re-arrange results in the same order as clients' cids impose
            ordered_results[int(fitres.metrics['cid'])] = (proxy, fitres)

        results = self.attack_fn(
            ordered_results, clients_state, magnitude=self.magnitude,
            w_re=self.aggregated_parameters, malicious_selected=self.malicious_selected,
            threshold=1e-5, d=params.shape[0]
        )

        if server_round >= self.warmup_rounds:
            M = load_all_time_series(dir="/Users/eddie/Documents/Università/ComputerScience/Thesis/flwr-pytorch/main/clients_params")
            M = np.transpose(M, (0, 2, 1))
            M_hat = M[:,:,-1].copy()
            pred_step = 1
            Mr = self.mar(M[:,:,:-1], pred_step)
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

            parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

            # For clients detected as malicious, set their parameters to be the averaged ones in their files
            # otherwise the forecasting in next round won't be reliable
            for idx in malicious_clients_idx:
                params = load_time_series(dir="/Users/eddie/Documents/Università/ComputerScience/Thesis/flwr-pytorch/main/clients_params", cid=idx)
                save_params(flatten_params(parameters_to_ndarrays(parameters_aggregated)), idx, remove_last=True)
        else:
            parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        return parameters_aggregated, metrics_aggregated

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

    def mar(self, X, pred_step, maxiter = 100):
        m, n, T = X.shape
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
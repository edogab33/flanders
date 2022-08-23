import flwr as fl
import numpy as np
import os

from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union
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


class FedMSCRED(fl.server.strategy.FedAvg):
    """Configurable MaliciousFedAvg strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        fraction_malicious: float = 0.0,
        magnitude: float = 1.0,
        threshold: float = 0.005,
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
        self.aggr_losses = np.array([])
        self.m = []                                              # number of malicious clients (updates each round)
        self.sample_size = []                                    # number of clients available (updates each round)
    
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

        self.m.append(int(sample_size * self.fraction_malicious))

        print("sample size: "+str(sample_size))
        print("num m: "+str(self.m[-1]))

        fit_ins_array = [
            FitIns(parameters, dict(config, **{"malicious": True, "magnitude": self.magnitude}) if idx < self.m[-1] else dict(config, **{"malicious": False}))
            for idx,_ in enumerate(clients)]

        return [(client, fit_ins_array[idx]) for idx,client in enumerate(clients)]

    def aggregate_fit(
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

        weights_results = {
            proxy.cid: np.asarray(parameters_to_ndarrays(fit_res.parameters))
            for proxy, fit_res in results
        }
        
        params = np.asarray([])
        for cid in weights_results:
            print("weights of " + str(cid)+ " " + str(weights_results[cid].shape))
            weights = weights_results[cid][0].reshape((*weights_results[cid][0].shape[:-2], -1))[:200]
            
            # Instead of making time series of TxN make them CxN where C is the number of clients
            params = np.vstack((params, weights)) if params.size else weights
            print("params at " + str(cid)+ " " + str(params.shape))

        # For each client, make signature test matrices
        # TODO: don't build again previously built matrices if they already exist
        mg.generate_train_test_data(params_time_series=np.transpose(params), test_end=params.shape[0], step_max=1)

        # Load MSCRED trained model and generate reconstructed matrices
        mg.generate_reconstructed_matrices(test_end_id=params.shape[0], sensor_n=params.shape[1], step_max=1)

        # Check if reconstucted matrices have errors above the threshold
        anomaly = eval.evaluate(test_end_point=params.shape[0], threshold=self.threshold)

        # TODO: If any signature is malicious, exclude the client from the average
        if anomaly:
            # new round
            print("ANOMALY FOUND")
            #parameters_aggregated = ndarrays_to_parameters(parameters_aggregated)
        #else:
        #    parameters_aggregated = ndarrays_to_parameters(parameters_aggregated)

        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        return parameters_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        config = {"strategy": "FedMSCRED", "fraction_mal": self.fraction_malicious, "magnitude": self.magnitude, 
            "frac_fit": self.fraction_fit, "frac_eval": self.fraction_evaluate, "min_fit_clients": self.min_fit_clients,
            "min_eval_clients": self.min_evaluate_clients, "min_available_clients": self.min_available_clients,
            "num_clients": self.sample_size, "num_malicious": self.m}
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
        np.save("results/aggregated_losses.npy", self.aggr_losses)

        return loss_aggregated, metrics_aggregated
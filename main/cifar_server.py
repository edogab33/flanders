import argparse
from multiprocessing import pool
import shutil
import flwr as fl
from flwr.common.typing import Scalar
import torch
import numpy as np
import os
import pandas as pd

from typing import Dict, Callable, Optional, Tuple, List
from neural_networks.dataset_utils import get_mnist, do_fl_partitioning, get_dataloader, get_circles, get_cifar_10, get_partitioned_income
from neural_networks.neural_networks import MnistNet, ToyNN, test_toy, train_mnist, test_mnist, train_toy
from clients import CifarClient, IncomeClient, ToyClient, set_params, get_params, MnistClient, set_sklearn_model_params, get_sklearn_model_params
from neural_networks.neural_networks import CifarNet, test_cifar
from strategy.utilities import save_results

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from strategy.fltrust import FLTrust
from strategy.malicious_fedavg import MaliciousFedAvg
from strategy.fedmedian import FedMedian
from strategy.flanders_local import LocalFlanders
from strategy.flanders_global import GlobalFlanders
from strategy.krum import Krum
from strategy.multikrum import MultiKrum
from strategy.trimmedmean import TrimmedMean
from strategy.generate_dataset_fg import GenerateDataset

from attacks import fang_attack, gaussian_attack, lie_attack, no_attack, pga_attack

from flwr.server.strategy.fedavg import FedAvg
from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")

parser.add_argument("--num_client_cpus", type=int, default=1)
parser.add_argument("--num_rounds", type=int, default=100)

def fit_config(server_round: int) -> Dict[str, Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": 1,  # number of local epochs
        "batch_size": 32,
    }
    return config

def mnist_evaluate(
    server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
):
    """Use the entire MNIST test set for evaluation."""

    # determine device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = MnistNet()
    set_params(model, parameters)
    model.to(device)

    testset = MNIST("", train=False, download=True, transform=transforms.ToTensor())
    testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=1)
    loss, accuracy = test_mnist(model, testloader, device=device)

    config["round"] = server_round
    save_results(loss, accuracy, config=config)

    # return statistics
    return loss, {"accuracy": accuracy}

def cifar_evaluate(
    server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
) -> Optional[Tuple[float, float]]:
    """Use the entire CIFAR-10 test set for evaluation."""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = CifarNet()
    set_params(model, parameters)
    model.to(device)

    _, testset = get_cifar_10()
    testloader = torch.utils.data.DataLoader(testset, batch_size=50)
    loss, accuracy = test_cifar(model, testloader, device=device)

    config["round"] = server_round
    save_results(loss, accuracy, config=config)

    # return statistics
    return loss, {"accuracy": accuracy}

def income_evaluate(
    server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
) -> Optional[Tuple[float, float]]:
    """Use the entire Income test set for evaluation."""
    model = LogisticRegression()
    set_sklearn_model_params(model, parameters)
    model.classes_ = np.array([0.0, 1.0])

    _, x_test, _, y_test = get_partitioned_income("neural_networks/adult.csv", pool_size)
    x_test = x_test[0]
    y_test = y_test[0]
    print("x_test ", x_test.shape)
    y_pred = model.predict(x_test)
    #accuracy = model.score(x_test, y_test)
    accuracy = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, model.predict_proba(x_test))

    config["round"] = server_round
    save_results(loss, accuracy, config=config)

    return loss, {"accuracy": accuracy}


def circles_evaluate(
    server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
):
    # determine device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = ToyNN()
    set_params(model, parameters)
    model.to(device)

    testloader = get_circles(32, n_samples=10000, workers=1, is_train=False)
    loss, accuracy = test_toy(model, testloader, device=device)

    config["round"] = server_round
    save_results(loss, accuracy, config=config)

    # return statistics
    return loss, {"accuracy": accuracy}

if __name__ == "__main__":

    # parse input arguments
    args = parser.parse_args()

    pool_size = 6  # number of dataset partions (= number of total clients)
    client_resources = {
        "num_cpus": args.num_client_cpus
    }  # each client will get allocated 1 CPUs

    # configure the strategy
    strategy = MultiKrum(
        fraction_fit=1,
        fraction_evaluate=0,                # no federated evaluation
        malicious_clients=2,
        min_fit_clients=5,
        min_evaluate_clients=0,
        magnitude=20,
        warmup_rounds=3,                    # Used only in GlobalFlanders
        to_keep=8,
        threshold=0.005,
        min_available_clients=pool_size,    # All clients should be available
        on_fit_config_fn=fit_config,
        evaluate_fn=income_evaluate,    # centralised evaluation of global model
        attack_fn=fang_attack,
        attack_name="no attack",
        strategy_name="fedavg",
        dataset_name="income",
        #initial_parameters=initial_parameters
    )

    # Cifar-10 dataset
    #train_path, testset = get_cifar_10()
    #fed_dir = do_fl_partitioning(
    #    train_path, pool_size=pool_size, alpha=1000, num_classes=10, val_ratio=0.1
    #)
    
    # Income dataset
    X_train, X_test, y_train, y_test = get_partitioned_income("neural_networks/adult.csv", pool_size)

    def client_fn(cid: int):
        cid = int(cid)
        # create a single client instance
        return IncomeClient(cid, X_train[cid], y_train[cid], X_test[cid], y_test[cid])
        #return ToyClient(cid, pool_size)

    # (optional) specify Ray config
    ray_init_args = {"include_dashboard": False}

    # Prepare directory for logs
    dirs = [f for f in os.listdir("results/") if not f.startswith('.')]

    # find the highest number in a list composed by strings that have a number as final char
    longest_string = len(max(dirs, key=len))
    idx = -2 if longest_string > 5 else -1

    highest_number = max([int(x[idx:]) for x in dirs if x[idx:].isdigit()])
    os.makedirs("results/run_"+str(highest_number+1), exist_ok=True)

    # Delete previous tensor in client_params
    tensor_dir = "clients_params/"
    if os.path.exists(tensor_dir):
        shutil.rmtree(tensor_dir)

    # start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=pool_size,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        ray_init_args=ray_init_args,
    )

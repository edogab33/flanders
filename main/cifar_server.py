import argparse
import shutil
import flwr as fl
from flwr.common.typing import Scalar
import torch
import numpy as np
import os
import pandas as pd

from typing import Dict, Callable, Optional, Tuple, List
from neural_networks.dataset_utils import get_mnist, do_fl_partitioning, get_dataloader, get_circles, get_cifar_10, get_partitioned_income
from neural_networks.neural_networks import MnistNet, ToyNN, roc_auc_multiclass, test_toy, train_mnist, test_mnist, train_toy
from clients import CifarClient, IncomeClient, ToyClient, set_params, get_params, MnistClient, set_sklearn_model_params, get_sklearn_model_params
from neural_networks.neural_networks import CifarNet, test_cifar
from strategy.utilities import save_results

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from strategy.fltrust import FLTrust
from strategy.malicious_fedavg import MaliciousFedAvg
from strategy.fedmedian import FedMedian
from strategy.flanders_global import GlobalFlanders
from strategy.krum import Krum
from strategy.multikrum import MultiKrum
from strategy.trimmedmean import TrimmedMean
from strategy.generate_dataset_fg import GenerateDataset

from attacks import fang_attack, gaussian_attack, lie_attack, no_attack, minmax_attack

from flwr.server.strategy.fedavg import FedAvg
from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import roc_auc_score


parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")

parser.add_argument("--num_client_cpus", type=int, default=1)
parser.add_argument("--num_rounds", type=int, default=100)
parser.add_argument("--exp_num", type=int, default=0)

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
    loss, accuracy, auc = test_mnist(model, testloader, device=device)

    config["round"] = server_round
    config["auc"] = auc
    save_results(loss, accuracy, config=config)

    # return statistics
    return loss, {"accuracy": accuracy, "auc": auc}

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
    loss, accuracy, auc = test_cifar(model, testloader, device=device)

    config["round"] = server_round
    config["auc"] = auc
    save_results(loss, accuracy, config=config)

    # return statistics
    return loss, {"accuracy": accuracy, "auc": auc}

def income_evaluate(
    server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
) -> Optional[Tuple[float, float]]:
    """Use the entire Income test set for evaluation."""
    model = LogisticRegression()
    model = set_sklearn_model_params(model, parameters)
    model.classes_ = np.array([0.0, 1.0])

    _, x_test, _, y_test = get_partitioned_income("neural_networks/adult.csv", 1)
    x_test = x_test[0]
    y_test = y_test[0]
    y_pred = model.predict(x_test)
    #accuracy = model.score(x_test, y_test)
    accuracy = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, model.predict_proba(x_test))
    auc = roc_auc_score(y_test, model.predict_proba(x_test)[:,1])

    config["round"] = server_round
    config["auc"] = auc
    save_results(loss, accuracy, config=config)

    return loss, {"accuracy": accuracy, "auc": auc}


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
    
    config = pd.read_csv("experiments_config.csv")
    config = config.to_dict("records")[args.exp_num]
    print(config)
    pool_size = config.get("pool_size", 10)  # number of dataset partions (= number of total clients)
    fraction_fit = config.get("fraction_fit", 1)
    fraction_evaluate = config.get("fraction_evaluate", 0)
    malicious_clients = config.get("malicious_clients", 0)
    min_fit_clients = config.get("min_fit_clients", 10)
    min_evaluate_clients = config.get("min_evaluate_clients", 0)
    magnitude = config.get("magnitude", 0)
    warmup_rounds = config.get("warmup_rounds", 0)
    to_keep = config.get("to_keep", 0)
    threshold = config.get("threshold", 1e-5)
    attack_name = config.get("attack_name", "no attack")
    strategy_name = config.get("strategy_name", "fedavg")
    dataset_name = config.get("dataset_name", "circles")
    window = config.get("window", 0)

    if dataset_name == "circles":
        evaluate_fn = circles_evaluate
        client_func = ToyClient
    elif dataset_name == "mnist":
        evaluate_fn = mnist_evaluate
        client_func = MnistClient
    elif dataset_name == "cifar":
        evaluate_fn = cifar_evaluate
        client_func = CifarClient
    elif dataset_name == "income":
        evaluate_fn = income_evaluate
        client_func = IncomeClient
    elif dataset_name == "houses":
        pass

    if attack_name == "no attack":
        attack_fn = no_attack
    elif attack_name == "gaussian":
        attack_fn = gaussian_attack
    elif attack_name == "lie":
        attack_fn = lie_attack
    elif attack_name == "fang":
        attack_fn = fang_attack
    elif attack_name == "minmax":
        attack_fn = minmax_attack

    if strategy_name == "avg":
        strategy_fn = MaliciousFedAvg
    elif strategy_name == "median":
        strategy_fn = FedMedian
    elif strategy_name == "trimmedmean":
        strategy_fn = TrimmedMean
    elif strategy_name == "krum":
        strategy_fn = Krum
    elif strategy_name == "multikrum":
        strategy_fn = MultiKrum
    elif strategy_name == "fltrust":
        strategy_fn = FLTrust
    elif strategy_name == "flanders":
        strategy_fn = GlobalFlanders
    elif strategy_name == "bulyan":
        pass

    client_resources = {
        "num_cpus": args.num_client_cpus
    }  # each client will get allocated 1 CPUs

    # configure the strategy
    strategy = strategy_fn(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,                # no federated evaluation
        malicious_clients=malicious_clients,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        magnitude=magnitude,
        warmup_rounds=warmup_rounds,                        # Used only in GlobalFlanders
        to_keep=to_keep,                                    # Used in Flanders and Krum/MultiKrum
        threshold=threshold,                                # 1e-5 for fang attack and minmax attacks. Used also in Flanders
        min_available_clients=pool_size,                    # All clients should be available
        on_fit_config_fn=fit_config,
        evaluate_fn=evaluate_fn,                            # centralised evaluation of global model
        attack_fn=attack_fn,
        attack_name=attack_name,                            # minmax, fang, gaussian, lie, no attack
        strategy_name=strategy_name,                        # avg, median, krum, multikrum, trimmedmean, fltrust, flanders
        dataset_name=dataset_name,                          # mnist, cifar, income, circles
        window=window,                                      # Used in Flanders
        #initial_parameters=initial_parameters
    )

    if dataset_name == "cifar":
        # Cifar-10 dataset
        train_path, testset = get_cifar_10()
        fed_dir = do_fl_partitioning(
            train_path, pool_size=pool_size, alpha=1000, num_classes=10, val_ratio=0.1
        )
    elif dataset_name == "income":
        # Income dataset
        X_train, X_test, y_train, y_test = get_partitioned_income("neural_networks/adult.csv", pool_size)

    def client_fn(cid: int):
        cid = int(cid)
        # create a single client instance
        if dataset_name == "cifar":
            return CifarClient(cid, fed_dir)
        elif dataset_name == "income":
            return IncomeClient(cid, X_train[cid], y_train[cid], X_test[cid], y_test[cid])
        else:
            return client_func(cid, pool_size)

    # (optional) specify Ray config
    ray_init_args = {"include_dashboard": False}

    # Prepare directory for logs
    dirs = [f for f in os.listdir("results_graphs/") if not f.startswith('.')]
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

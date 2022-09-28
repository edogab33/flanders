import argparse
import shutil
import flwr as fl
from flwr.common.typing import Scalar
import ray
import torch
import torchvision
import numpy as np
import os
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Callable, Optional, Tuple, List
from cifar_nn.dataset_utils import get_mnist, do_fl_partitioning, get_dataloader, get_circles
from cifar_nn.utils import MnistNet, ToyNN, test_toy, train_mnist, test_mnist, train_toy

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
from strategy.generate_dataset_fg import GenerateDataset

from attacks import fang_attack, gaussian_attack, lie_attack, no_attack, pga_attack

from flwr.server.strategy.fedavg import FedAvg

from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from strategy.trimmedmean import TrimmedMean

parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")

parser.add_argument("--num_client_cpus", type=int, default=1)
parser.add_argument("--num_rounds", type=int, default=100)


# Flower client, adapted from Pytorch quickstart example
class MnistClient(fl.client.NumPyClient):
    def __init__(self, cid: str):
        self.cid = cid
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
        self.net = MnistNet()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_parameters(self, config):
        return get_params(self.net)

    def fit(self, parameters, config):
        set_params(self.net, parameters)

        # Load data for this client and get trainloader
        num_workers = len(ray.worker.get_resource_ids()["CPU"])

        trainloader = get_mnist("./cifar_nn/data", 32, self.cid, nb_clients=pool_size, is_train=True, workers=num_workers)

        # Send model to device
        self.net.to(self.device)

        # Train
        train_mnist(self.net, trainloader, epochs=config["epochs"], device=self.device)

        new_parameters = self.get_parameters(config={})

        #if "malicious" in config:
        #    if config["malicious"]:
        #        magnitude = config["magnitude"]
        #        # Add random perturbation.
        #        perturbate = lambda a: a + np.random.normal(loc=0, scale=magnitude, size=len(a))
        #        new_parameters = np.apply_along_axis(perturbate, 0, new_parameters).tolist()

        # Return local model and statistics
        return new_parameters, len(trainloader.dataset), {"malicious": config["malicious"], "cid": self.cid}

    def evaluate(self, parameters, config):
        set_params(self.net, parameters)

        # Load data for this client and get trainloader
        num_workers = len(ray.worker.get_resource_ids()["CPU"])
        testloader = get_mnist("./cifar_nn/data", 32, self.cid, nb_clients=pool_size, is_train=False, workers=num_workers)

        # Send model to device
        self.net.to(self.device)

        # Evaluate
        loss, accuracy = test_mnist(self.net, testloader, device=self.device)

        # Return statistics
        return float(loss), len(testloader), {"accuracy": float(accuracy)}

class ToyClient(fl.client.NumPyClient):
    def __init__(self, cid: str):
        self.cid = cid
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
        self.net = ToyNN()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_parameters(self, config):
        return get_params(self.net)

    def fit(self, parameters, config):
        set_params(self.net, parameters)

        # Load data for this client and get trainloader
        num_workers = len(ray.worker.get_resource_ids()["CPU"])
        trainloader = get_circles(32, n_samples=10000, workers=num_workers, is_train=True)

        # Send model to device
        self.net.to(self.device)

        # Train
        train_toy(self.net, trainloader, epochs=config["epochs"], device=self.device)

        new_parameters = self.get_parameters(config={})

        #if "malicious" in config:
        #    if config["malicious"]:
        #        magnitude = config["magnitude"]
        #        # Add random perturbation.
        #        perturbate = lambda a: a + np.random.normal(loc=0, scale=magnitude, size=len(a))
        #        new_parameters = np.apply_along_axis(perturbate, 0, new_parameters).tolist()

        # Return local model and statistics
        return new_parameters, len(trainloader.dataset), {"malicious": config["malicious"], "cid": self.cid}

    def evaluate(self, parameters, config):
        set_params(self.net, parameters)

        # Load data for this client and get trainloader
        num_workers = len(ray.worker.get_resource_ids()["CPU"])
        testloader = get_circles(32, n_samples=10000, workers=num_workers, is_train=False)

        # Send model to device
        self.net.to(self.device)

        # Evaluate
        loss, accuracy = test_toy(self.net, testloader, device=self.device)

        # Return statistics
        return float(loss), len(testloader), {"accuracy": float(accuracy)}

def fit_config(server_round: int) -> Dict[str, Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": 1,  # number of local epochs
        "batch_size": 32,
    }
    return config


def get_params(model: torch.nn.ModuleList) -> List[np.ndarray]:
    """Get model weights as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_params(model: torch.nn.ModuleList, params: List[np.ndarray]):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def mnist_evaluate_fn(
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

    save_results(loss, accuracy, config=config)

    # return statistics
    return loss, {"accuracy": accuracy}

def circles_evaluate_fn(
    server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
):
    # determine device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = ToyNN()
    set_params(model, parameters)
    model.to(device)

    testloader = get_circles(32, n_samples=10000, workers=1, is_train=False)
    loss, accuracy = test_toy(model, testloader, device=device)

    save_results(loss, accuracy, config=config)

    # return statistics
    return loss, {"accuracy": accuracy}

def save_results(loss, accuracy, config=None):
    # Save results as npy file
    dirs = [f for f in os.listdir("results/") if not f.startswith('.')]
    longest_string = len(max(dirs, key=len))
    idx = -2 if longest_string > 5 else -1

    highest_number = str(max([int(x[idx:]) for x in dirs if x[idx:].isdigit()]))
    loss_series = []
    acc_series = []
    loss_path = "results/run_"+highest_number+"/loss.npy"
    acc_path = "results/run_"+highest_number+"/acc.npy"
    if os.path.exists(loss_path):
        loss_series = np.load(loss_path)
    if os.path.exists(acc_path):
        acc_series = np.load(acc_path)
    loss_series = np.save(loss_path, np.append(loss_series, loss))
    acc_series = np.save(acc_path, np.append(acc_series, accuracy))

    # Save config
    config_path = "results/run_"+highest_number+"/config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)

if __name__ == "__main__":

    # parse input arguments
    args = parser.parse_args()

    pool_size = 10  # number of dataset partions (= number of total clients)
    client_resources = {
        "num_cpus": args.num_client_cpus
    }  # each client will get allocated 1 CPUs

    # configure the strategy
    strategy = Krum(
        fraction_fit=1,
        fraction_evaluate=0,                # no federated evaluation
        malicious_clients=5,
        min_fit_clients=10,
        min_evaluate_clients=0,
        magnitude=20,
        warmup_rounds=3,
        to_keep=8,
        threshold=0.005,
        min_available_clients=pool_size,    # All clients should be available
        on_fit_config_fn=fit_config,
        evaluate_fn=circles_evaluate_fn,    # centralised evaluation of global model
        attack_fn=fang_attack,
        #initial_parameters=initial_parameters
    )

    def client_fn(cid: int):
        # create a single client instance
        return MnistClient(cid)

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

import argparse
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
from cifar_nn.dataset_utils import get_mnist, do_fl_partitioning, get_dataloader
from cifar_nn.utils import Net, train, test

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from strategy.malicious_fedavg import MaliciousFedAvg
from strategy.fedmedian import FedMedian
from strategy.flanders_local import LocalFlanders
from strategy.flanders_global import GlobalFlanders
from strategy.krum import Krum
from strategy.multikrum import MultiKrum
from strategy.generate_dataset_fg import GenerateDataset
from flwr.server.strategy.fedavg import FedAvg

parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")

parser.add_argument("--num_client_cpus", type=int, default=1)
parser.add_argument("--num_rounds", type=int, default=5)


# Flower client, adapted from Pytorch quickstart example
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: str):
        self.cid = cid
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}

        # Instantiate model
        self.net = Net()

        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_parameters(self, config):
        return get_params(self.net)

    def fit(self, parameters, config):
        set_params(self.net, parameters)

        # Load data for this client and get trainloader
        num_workers = len(ray.worker.get_resource_ids()["CPU"])

        trainloader = get_mnist("./cifar_nn/data", 64, self.cid, nb_clients=pool_size, is_train=True, workers=num_workers)

        # Send model to device
        self.net.to(self.device)

        # Train
        train(self.net, trainloader, epochs=config["epochs"], device=self.device)

        new_parameters = self.get_parameters(config={})

        print("CLIENT CONFIG "+str(config))

        if "malicious" in config:
            if config["malicious"]:
                magnitude = config["magnitude"]
                # Add random perturbation.
                perturbate = lambda a: a + np.random.normal(loc=0, scale=magnitude, size=len(a))
                new_parameters = np.apply_along_axis(perturbate, 0, new_parameters).tolist()

        # Return local model and statistics
        return new_parameters, len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_params(self.net, parameters)

        # Load data for this client and get trainloader
        num_workers = len(ray.worker.get_resource_ids()["CPU"])
        testloader = get_mnist("./cifar_nn/data", 64, self.cid, nb_clients=pool_size, is_train=False, workers=num_workers)

        # Send model to device
        self.net.to(self.device)

        # Evaluate
        loss, accuracy = test(self.net, testloader, device=self.device)

        # Return statistics
        return float(loss), len(testloader), {"accuracy": float(accuracy)}


def fit_config(server_round: int) -> Dict[str, Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": 1,  # number of local epochs
        "batch_size": 64,
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


def evaluate_fn(
    erver_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
):
    """Use the entire CIFAR-10 test set for evaluation."""

    # determine device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net()
    set_params(model, parameters)
    model.to(device)

    testset = MNIST("", train=False, download=True, transform=transforms.ToTensor())
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=1)
    loss, accuracy = test(model, testloader, device=device)

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

    # return statistics
    return loss, {"accuracy": accuracy}


# Start simulation (a _default server_ will be created)
# This example does:
# 1. Downloads CIFAR-10
# 2. Partitions the dataset into N splits, where N is the total number of
#    clients. We refere to this as `pool_size`. The partition can be IID or non-IID
# 3. Starts a simulation where a % of clients are sample each round.
# 4. After the M rounds end, the global model is evaluated on the entire testset.
#    Also, the global model is evaluated on the valset partition residing in each
#    client. This is useful to get a sense on how well the global model can generalise
#    to each client's data.
if __name__ == "__main__":

    # parse input arguments
    args = parser.parse_args()

    pool_size = 10  # number of dataset partions (= number of total clients)
    client_resources = {
        "num_cpus": args.num_client_cpus
    }  # each client will get allocated 1 CPUs

    # configure the strategy
    strategy = GlobalFlanders(
        fraction_fit=1,
        fraction_evaluate=1,
        fraction_malicious=0.0,
        min_fit_clients=10,
        min_evaluate_clients=10,
        magnitude=0,
        min_available_clients=pool_size,  # All clients should be available
        on_fit_config_fn=fit_config,
        evaluate_fn=evaluate_fn,  # centralised evaluation of global model
    )

    def client_fn(cid: int):
        # create a single client instance
        return FlowerClient(cid)

    # (optional) specify Ray config
    ray_init_args = {"include_dashboard": False}

    # Prepare directory for logs
    dirs = [f for f in os.listdir("results/") if not f.startswith('.')]

    # find the highest number in a list composed by strings that have a number as final char
    longest_string = len(max(dirs, key=len))
    idx = -2 if longest_string > 5 else -1

    highest_number = max([int(x[idx:]) for x in dirs if x[idx:].isdigit()])
    os.makedirs("results/run_"+str(highest_number+1), exist_ok=True)

    # start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=pool_size,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        ray_init_args=ray_init_args,
    )

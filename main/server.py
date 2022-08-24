from unittest import result
import flwr as fl
from client import FlowerClient
from strategy.malicious_fedavg import MaliciousFedAvg
from strategy.fedmedian import FedMedian
from strategy.fedmscred import FedMSCRED
from strategy.fedmscred2 import FedMSCRED2
from strategy.krum import Krum
from flwr.server.strategy.fedavg import FedAvg
import torch
import pytorch_lightning as pl
import mnist
from collections import OrderedDict
import numpy as np
import os
import json

def fit_config(server_round: int):
    config = {
        "current_round": server_round,
        "local_epochs": 1,
    }
    return config

def client_fn(cid: str) -> FlowerClient:
    model = mnist.LitMNIST()
    train_loader, val_loader, test_loader = mnist.load_data()
    return FlowerClient(model, train_loader, val_loader, test_loader)

def evaluate_fn(server_round, parameters, config):
    model = mnist.LitMNIST(client=False)
    _, _, test_loader = mnist.load_data(client=False)

    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

    trainer = pl.Trainer(max_epochs=1)
    if torch.cuda.is_available():
        trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=1)

    results = trainer.test(model, test_loader)

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
    loss_series = np.save(loss_path, np.append(loss_series, results[0]["test_loss"]))
    acc_series = np.save(acc_path, np.append(acc_series, results[0]["test_acc"]))

    # Save config
    config_path = "results/run_"+highest_number+"/config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)

    return [results[0]["test_loss"], {"accuracy": results[0]["test_acc"]}]

def main() -> None:
    # Define strategy
    strategy = FedMSCRED(
        fraction_fit=0.5,
        fraction_evaluate=0.0,                           # set 0 to disable client evaluation
        evaluate_fn=evaluate_fn,
        fraction_malicious=0.4,                          # computed from the number of available clients
        magnitude=10.0,
        min_available_clients=5,
        min_fit_clients=5,
        min_evaluate_clients=5,
        #threshold=0.005,
    )

    # Prepare directory for logs
    dirs = [f for f in os.listdir("results/") if not f.startswith('.')]

    # find the highest number in a list composed by strings that have a number as final char
    longest_string = len(max(dirs, key=len))
    idx = -2 if longest_string > 5 else -1
    highest_number = max([int(x[idx:]) for x in dirs if x[idx:].isdigit()])
    os.makedirs("results/run_"+str(highest_number+1), exist_ok=True)

    #fl.simulation.start_simulation(
    #    client_fn=client_fn,
    #    num_clients=10,
    #    config=fl.server.ServerConfig(num_rounds=50),
    #    strategy=strategy,
    #)
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=50),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
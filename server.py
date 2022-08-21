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
    print("RESULTS:")
    print(results)
    return [results[0]["test_loss"], {"accuracy": results[0]["test_acc"]}]

def main() -> None:
    # Define strategy
    strategy = Krum(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        evaluate_fn=evaluate_fn,
        fraction_malicious=0.0,                          # computed from the number of available clients
        magnitude=20.0,
        #threshold=0.005,
    )
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=5,
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )
    #fl.server.start_server(
    #    server_address="0.0.0.0:8080",
    #    config=fl.server.ServerConfig(num_rounds=2),
    #    strategy=strategy
    #)

if __name__ == "__main__":
    main()
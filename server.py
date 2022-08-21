import flwr as fl
from client import FlowerClient
from strategy.malicious_fedavg import MaliciousFedAvg
from strategy.fedmedian import FedMedian
from strategy.fedmscred import FedMSCRED
from strategy.fedmscred2 import FedMSCRED2
from strategy.krum import Krum
from flwr.server.strategy.fedavg import FedAvg
import mnist

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

def main() -> None:
    # Define strategy
    strategy = Krum(
        fraction_fit=0.2,
        fraction_evaluate=0.2,
        fraction_malicious=0.5,                          # computed from the number of available clients
        magnitude=20.0,
        #threshold=0.005,
    )
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=2,
        config=fl.server.ServerConfig(num_rounds=2),
        strategy=strategy,
    )
    #fl.server.start_server(
    #    server_address="0.0.0.0:8080",
    #    config=fl.server.ServerConfig(num_rounds=2),
    #    strategy=strategy
    #)

if __name__ == "__main__":
    main()
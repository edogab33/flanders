import flwr as fl
from strategy.malicious_fedavg import MaliciousFedAvg
from strategy.fedmedian import FedMedian
from strategy.fedmscred import FedMSCRED
from strategy.fedmscred2 import FedMSCRED2
from flwr.server.strategy.fedavg import FedAvg

def fit_config(server_round: int):
    config = {
        "current_round": server_round,
        "local_epochs": 1,
    }
    return config

def main() -> None:
    # Define strategy
    strategy = FedMSCRED2(
        fraction_fit=0.2,
        fraction_evaluate=0.2,
        fraction_malicious=0.5,                          # computed from the number of available clients
        magnitude=2.0,
        threshold=0.005,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=6),
        strategy=strategy
    )

if __name__ == "__main__":
    main()
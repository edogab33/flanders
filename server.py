import flwr as fl
from strategy.malicious_fedavg import MaliciousFedAvg

def fit_config(server_round: int):
    config = {
        "current_round": server_round,
        "local_epochs": 1,
    }
    return config

def main() -> None:
    # Define strategy
    strategy = MaliciousFedAvg(
        fraction_fit=0.2,
        fraction_evaluate=0.2,
        fraction_malicious=0.5                          # computed from the number of available clients
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy
    )

if __name__ == "__main__":
    main()
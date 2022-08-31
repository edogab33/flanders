from ast import List
import flwr as fl
import mnist
import pytorch_lightning as pl
from collections import OrderedDict
import numpy as np
import torch
import os

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_parameters(self, config):
        print("GET PARAMETERS")
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        print("SET PARAMETERS")
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print("I AM HERE - FIT")
        self.set_parameters(parameters)
        self.model.to(self.device)
        trainer = pl.Trainer(logger=False, enable_progress_bar=False, max_epochs=1)
        if torch.cuda.is_available():
            trainer = pl.Trainer(accelerator="gpu", devices=1, logger=False, enable_progress_bar=False, max_epochs=1)
        trainer.fit(self.model, self.train_loader, self.val_loader)

        new_parameters = self.get_parameters(config={})

        print("CLIENT CONFIG "+str(config))

        #np.save("/Users/eddie/Documents/Università/ComputerScience/Thesis/flwr-pytorch/params.npy", new_parameters[0])
        if "malicious" in config:
            if config["malicious"]:
                magnitude = config["magnitude"]
                # given a list of tensors of variable shape, return the first tensor with shape (64, 768) and add a random perturbation to it.
                perturbate = lambda a: a + np.random.normal(loc=0, scale=magnitude, size=len(a))
                new_parameters = np.apply_along_axis(perturbate, 0, new_parameters).tolist()

        return new_parameters, len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        
        trainer = pl.Trainer(logger=False, enable_progress_bar=False)
        if torch.cuda.is_available():
            trainer = pl.Trainer(accelerator="gpu", devices=1, logger=False, enable_progress_bar=False)
        results = trainer.test(self.model, self.test_loader)
        #print("RESULTS:")
        #print(results)
        loss = results[0]["cl_test_loss"]

        print("Client loss "+str(loss))

        return loss, 10000, {"loss": loss}


def main() -> None:
    # Model and data
    model = mnist.LitMNIST()
    train_loader, val_loader, test_loader = mnist.load_data()

    # Flower client
    client = FlowerClient(model, train_loader, val_loader, test_loader)
    fl.client.start_numpy_client(server_address="[::]:8080", client=client)


if __name__ == "__main__":
    main()

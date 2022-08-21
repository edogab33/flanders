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

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        trainer = pl.Trainer(max_epochs=1)
        if torch.cuda.is_available():
            trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=1)

        trainer.fit(self.model, self.train_loader, self.val_loader)

        new_parameters = self.get_parameters(config={})

        print("CLIENT CONFIG "+str(config))

        #np.save("/Users/eddie/Documents/Università/ComputerScience/Thesis/flwr-pytorch/params.npy", new_parameters[0])
        if "malicious" in config:
            if config["malicious"]:
                magnitude = config["magnitude"]
                # given a list of tensors of variable shape, return the first tensor with shape (64, 768) and add a random perturbation to it.
                #for tensor in tensors:
                #    if tensor.shape == (64, 768):
                #        return tensor + np.random.normal(loc=0, scale=magnitude, size=len(tensor))
                #return tensor
                perturbate = lambda a: a + np.random.normal(loc=0, scale=magnitude, size=len(a))
                new_parameters[0] = np.apply_along_axis(perturbate, 0, new_parameters[0])
                print(new_parameters[0].shape)

        # TODO: clients should be distinguished by their id (or something else): 
        # Save the weights in a file called "client_[x]_weights.npy" inside "strategy/clients_weights/".
        # Then the strategy can load the weights of all clients and append them in "histories/weights_[x]_history.npy".
        # check if strategy/clients_weights/ exists
        # --> NOT GOOD: in our threat model we cannot distinguish clients (they're malicious). 
        # Probably the best solution is to aggregate on the server, append to the aggregated history
        # and test against the trained model. If something is wrong, repeat the round.
        
        #if not os.path.exists("strategy/clients_weights"):
        #    os.makedirs("strategy/clients_weights")
        
        return new_parameters, 55000, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        
        trainer = pl.Trainer(max_epochs=1)
        if torch.cuda.is_available():
            trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=1)
        results = trainer.test(self.model, self.test_loader)
        loss = results[0]["test_loss"]

        print("Loss "+str(loss))

        return loss, 10000, {"loss": loss}


def main() -> None:
    # Model and data
    model = mnist.LitMNIST()
    train_loader, val_loader, test_loader = mnist.load_data()

    # Flower client
    client = FlowerClient(model, train_loader, val_loader, test_loader)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()
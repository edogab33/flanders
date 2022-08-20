import numpy as np
import os
from flwr.common import Parameters

def save_params(parameters_aggregated: Parameters, server_round: int):
    # TODO: test that:
    if parameters_aggregated is not None:
        # Save weights
        print("Loading previous weights...")
        # check file exists
        if os.path.isfile("strategy/util_histories/results.npy"):
            old_params = np.load("strategy/util_histories/weights.npy")
        else:
            old_params = np.array([])
        print(f"Saving round "+ str(server_round) +" weights...")
        new_params = np.vstack((old_params, parameters_aggregated.tensors))
        print("new param: "+str(new_params.shape))
        np.save(f"strategy/util_histories/round-"+str(server_round)+"-weights.npy", new_params)
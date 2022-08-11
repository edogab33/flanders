import numpy as np
from flwr.common import Parameters

def save_params(parameters_aggregated: Parameters, server_round: int):
    # TODO: test that:
    if parameters_aggregated is not None:
        # Save weights
        print("Loading previous weights...")
        old_params = np.load("weights.npy")
        print(f"Saving round "+ str(server_round) +" weights...")
        new_params = np.vstack((old_params, parameters_aggregated[0]))
        print("new param: "+str(new_params.shape))
        np.save(f"round-"+str(server_round)+"-weights.npy", new_params)
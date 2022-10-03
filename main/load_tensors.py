from strategy.utilities import load_all_time_series
import numpy as np
import matplotlib.pyplot as plt

tensor = load_all_time_series(dir="/Users/eddie/Documents/Università/ComputerScience/Thesis/flwr-pytorch/main/clients_params")

print(tensor.shape)
tensor = np.transpose(tensor, (1, 0, 2))
tensor = tensor.reshape(*tensor.shape[:-2], -1)
print(tensor.shape)

plt.plot(tensor[:])
plt.show()

from strategy.utilities import load_all_time_series
import numpy as np
import matplotlib.pyplot as plt

def mar(X, pred_step, maxiter = 100, window = 0):
    m, n, T = X.shape
    if window > 0:
        start = T - window
        start=0
        print("window: ", window)
        print("Start: ", start)
    B = np.random.randn(n, n)
    for it in range(maxiter):
        temp0 = B.T @ B
        print(temp0)
        temp1 = np.zeros((m, m))
        temp2 = np.zeros((m, m))
        for t in range(start, T):
            temp1 += X[:, :, t] @ B @ X[:, :, t - 1].T
            temp2 += X[:, :, t - 1] @ temp0 @ X[:, :, t - 1].T
        A = temp1 @ np.linalg.inv(temp2)
        temp0 = A.T @ A
        temp1 = np.zeros((n, n))
        temp2 = np.zeros((n, n))
        for t in range(start, T):
            temp1 += X[:, :, t].T @ A @ X[:, :, t - 1]
            temp2 += X[:, :, t - 1].T @ temp0 @ X[:, :, t - 1]
        B = temp1 @ np.linalg.inv(temp2)
    tensor = np.append(X, np.zeros((m, n, pred_step)), axis = 2)
    for s in range(pred_step):
        tensor[:, :, T + s] = A @ tensor[:, :, T + s - 1] @ B.T
    return tensor[:, :, - pred_step :]

tensor = load_all_time_series(dir="/Users/eddie/Documents/Università/ComputerScience/Thesis/flwr-pytorch/main/clients_params", window=10)
tensor = np.transpose(tensor, (0, 2, 1))
print(tensor.shape)

Mr = mar(tensor[:,:,:-1], 1, maxiter=100, window=9)
#print(Mr)
#for i in range(len(tensor)):
#    print(tensor[i])
tensor = np.transpose(tensor, (1, 0, 2))
tensor = tensor.reshape(*tensor.shape[:-2], -1)

#plt.plot(tensor[:])
#plt.show()

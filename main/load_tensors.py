from cmath import log
from strategy.utilities import load_all_time_series
import numpy as np
import matplotlib.pyplot as plt

def mar(X, pred_step, maxiter = 100, window = 0):
    print("ALS iterations: ", maxiter)
    m, n, T = X.shape
    if window > 0:
        start = T - window
    try:
        B = np.random.randn(n, n)
        for it in range(maxiter):
            temp0 = B.T @ B
            temp1 = np.zeros((m, m))
            temp2 = np.zeros((m, m))
            for t in range(start, T):
                temp1 += X[:, :, t] @ B @ X[:, :, t - 1].T
                temp2 += X[:, :, t - 1] @ temp0 @ X[:, :, t - 1].T
            #print(temp2)
            A = temp1 @ np.linalg.inv(temp2)
            temp0 = A.T @ A
            temp1 = np.zeros((n, n))
            temp2 = np.zeros((n, n))
            for t in range(start, T):
                temp1 += X[:, :, t].T @ A @ X[:, :, t - 1]
                temp2 += X[:, :, t - 1].T @ temp0 @ X[:, :, t - 1]
            temp2 = cap_values(temp2)
            B = temp1 @ np.linalg.inv(temp2)
        tensor = np.append(X, np.zeros((m, n, pred_step)), axis = 2)
        for s in range(pred_step):
            tensor[:, :, T + s] = A @ tensor[:, :, T + s - 1] @ B.T
        if np.isnan(tensor).any():
            raise ValueError("NaN values in tensor")
        return tensor[:, :, - pred_step :]
    except:
        tensor = mar(X, pred_step, maxiter = int(maxiter*0.75), window = window)
        return tensor

def cap_values(matrix):
    """
    Cap values of matrices in order to avoid
    hitting the limit of the floating point precision
    and to avoid singular matrices
    """
    #matrix = np.nan_to_num(matrix, nan=np.finfo(np.float64).max)
    matrix[matrix < np.finfo(np.float64).tiny] = np.finfo(np.float64).tiny
    return matrix

tensor = load_all_time_series(dir="/Users/eddie/Documents/Università/ComputerScience/Thesis/flwr-pytorch/main/clients_params")
tensor = np.transpose(tensor, (0, 2, 1))
print(tensor.shape)
M_hat = tensor[:,:,-1].copy()
Mr = mar(tensor[:,:,:-1], 1, maxiter=3000, window=4)
#print(Mr)
delta = np.subtract(M_hat, Mr[:,:,0])
anomaly_scores = np.sum(np.abs(delta)**2,axis=-1)**(1./2)
good_clients_idx = sorted(np.argsort(anomaly_scores)[:3])
print("Anomaly scores: ", anomaly_scores)
print("Kept clients: ")
print(good_clients_idx)
#for i in range(len(tensor)):
#    print(tensor[i])
#tensor = np.transpose(tensor, (1, 0, 2))
#tensor = tensor.reshape(*tensor.shape[:-2], -1)

#plt.plot(tensor[:])
#plt.show()

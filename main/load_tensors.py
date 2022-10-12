from cmath import log
from strategy.utilities import load_all_time_series, load_time_series
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

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
            temp2 = cap_values(temp2)
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
        print("[!!] Error in MAR - decreasing number of iterations")
        if int(maxiter*0.5) == 0:
            raise ValueError("Could not find a solution for MAR.")
        return mar(X, pred_step, maxiter = int(maxiter*0.5), window = window)

def cap_values(matrix):
    """
    Cap values of matrices in order to avoid
    hitting the limit of the floating point precision
    and to avoid singular matrices
    """
    matrix = np.nan_to_num(matrix, nan=np.finfo(np.float64).max)
    matrix[matrix < np.finfo(np.float64).tiny] = np.finfo(np.float64).tiny
    return matrix

tensor = load_all_time_series(dir="/Users/eddie/Documents/Università/ComputerScience/Thesis/flwr-pytorch/main/clients_params")
#print(tensor.shape)
#tensor = np.transpose(tensor, (0, 2, 1))


#print(tensor.shape)
#M_hat = tensor[:,:,-1].copy()
#Mr = mar(tensor[:,:,:-1], 1, maxiter=50, window=29)
#print(Mr)
#delta = np.subtract(M_hat, Mr[:,:,0])
#anomaly_scores = np.sum(np.abs(delta)**2,axis=-1)**(1./2)
#good_clients_idx = sorted(np.argsort(anomaly_scores)[:5])
#print("Anomaly scores: ", anomaly_scores)
#print("Kept clients: ")
#print(good_clients_idx)

#fig, ax = plt.subplots(1,3, figsize=(10,5))
#ax[0].matshow(M_hat)
#ax[1].matshow(Mr[:,:,0])
#ax[2].matshow(delta)
#plt.show()

#tensor = np.transpose(tensor, (1, 0, 2))
#print(tensor)
#tensor = tensor.reshape(*tensor.shape[:-2], -1)
#max_value = np.max(tensor, axis=1)
#min_value = np.min(tensor, axis=1)
#tensor = (np.transpose(tensor) - min_value)/(max_value - min_value + 1e-6)
#print(tensor)

tensor = np.transpose(tensor, (1, 0, 2))
tensor = tensor.reshape(*tensor.shape[:-2], -1)
print(tensor.shape)
plt.plot(tensor[:])
plt.show()

#m = load_time_series(dir="/Users/eddie/Documents/Università/ComputerScience/Thesis/flwr-pytorch/main/clients_params", cid=4)
#print(m.shape)
#plt.plot(m[:])
#plt.show()
import os
import numpy as np
import pandas as pd

def generate_train_test_data(
    params_time_series="/Users/eddie/Documents/Università/ComputerScience/Thesis/flwr-pytorch/main/strategy/histories/history.csv",
    matrix_data_path="/Users/eddie/Documents/Università/ComputerScience/Thesis/MSCRED/matrix_data/",
    train_start=0,
    train_end=0,
    test_start=0,
    test_end=0,
    gap_time=1,
    step_max=5,
    win_size=[1]):

    # load params_time_series and transform it into a csv
    data = np.array(pd.read_csv(params_time_series, header = None), dtype=np.float64)
    print(data.shape)

    sensor_n = data.shape[0]

    # min-max normalization
    max_value = np.max(data, axis=1)
    min_value = np.min(data, axis=1)
    data = (np.transpose(data) - min_value)/(max_value - min_value + 1e-6)
    print(data.shape)

    data = np.transpose(data)
    print(data.shape)

    # Check if the path exists, if not, create it:
    if not os.path.exists(matrix_data_path):
        os.makedirs(matrix_data_path)
    #else:
    #    shutil.rmtree(matrix_data_path)
    #    os.makedirs(matrix_data_path)

    #multi-scale signature matix generation
    for w in range(len(win_size)):
        matrix_all = []
        win = win_size[w]
        print ("generating signature with window " + str(win) + "...")
        for t in range(test_start, test_end, gap_time):
            matrix_t = np.zeros((sensor_n, sensor_n))
            if t >= win_size[-1]-1:
                for i in range(sensor_n):
                    for j in range(i, sensor_n):
                        matrix_t[i][j] = np.inner(data[i, t - win:t], data[j, t - win:t])/(win) # rescale by win
                        matrix_t[j][i] = matrix_t[i][j]
            matrix_all.append(matrix_t)

        path_temp = matrix_data_path + "matrix_win_" + str(win)

        np.save(path_temp, matrix_all)

generate_train_test_data(test_start=0, test_end=400)
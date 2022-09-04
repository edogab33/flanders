from cgi import test
import matplotlib.pyplot as plt
import numpy as np
import os

def evaluate(
    threshold=0.005,
    gap_time=1,
    test_start_point=0,
    test_end_point=0,
    matrix_data_path='strategy/mscred/matrix_data/'):

    thred_b = threshold
    test_start = int(test_start_point/gap_time)
    test_end = int(test_end_point/gap_time)

    test_anomaly_score = np.zeros((test_end - test_start, 1))

    test_data_path = matrix_data_path + "test_data/"
    reconstructed_data_path = matrix_data_path + "reconstructed_data/"

    for i in range(test_start, test_end):
        path_temp_1 = os.path.join(test_data_path, "test_data_" + str(i) + '.npy')
        gt_matrix_temp = np.load(path_temp_1)

        path_temp_2 = os.path.join(reconstructed_data_path, "reconstructed_data_" + str(i) + '.npy')
        reconstructed_matrix_temp = np.load(path_temp_2)
        reconstructed_matrix_temp = np.transpose(reconstructed_matrix_temp[0], [0, 3, 1, 2])

        #first (short) duration scale for evaluation  
        select_gt_matrix = np.array(gt_matrix_temp)[-1][0] #get last step matrix

        select_reconstructed_matrix = np.array(reconstructed_matrix_temp)[0][0]

        #compute number of broken element in residual matrix
        select_matrix_error = np.square(np.subtract(select_gt_matrix, select_reconstructed_matrix))
        num_broken = len(select_matrix_error[select_matrix_error > thred_b])
        test_anomaly_score[i - test_start] = num_broken

    test_anomaly_score = test_anomaly_score.ravel()

    anomaly_scores = [test_anomaly_score[i] for i in range(len(test_anomaly_score))]
    return anomaly_scores






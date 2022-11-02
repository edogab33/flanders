from cgi import test
import matplotlib.pyplot as plt
import numpy as np
import os

def evaluate(
    threshold=0.005,
    gap_time=1,
    test_matrix_id=0,
    matrix_data_path='strategy/mscred/matrix_data/'
    ):
    '''
    Code adapted from: https://github.com/7fantasysz/MSCRED
    '''

    thred_b = threshold

    test_data_path = matrix_data_path + "test_data/"
    reconstructed_data_path = matrix_data_path + "reconstructed_data/"

    path_temp_1 = os.path.join(test_data_path, "test_data_" + str(test_matrix_id) + '.npy')
    gt_matrix_temp = np.load(path_temp_1)

    test_anomaly_score = np.zeros((gt_matrix_temp.shape[0], 1))

    path_temp_2 = os.path.join(reconstructed_data_path, "reconstructed_data_" + str(test_matrix_id) + '.npy')
    reconstructed_matrix_temp = np.load(path_temp_2)
    reconstructed_matrix_temp = np.transpose(reconstructed_matrix_temp[0], [0, 3, 1, 2])

    #first (short) duration scale for evaluation  
    select_gt_matrix = np.array(gt_matrix_temp)[-1][0] #get last step matrix

    select_reconstructed_matrix = np.array(reconstructed_matrix_temp)[0][0]

    #compute number of broken element in residual matrix
    select_matrix_error = np.square(np.subtract(select_gt_matrix, select_reconstructed_matrix))
    num_broken = len(select_matrix_error[select_matrix_error > thred_b])

    anomaly_scores = []
    #compute anomaly score for each client
    for client in select_matrix_error:
        anomaly_scores.append(np.sum(client))

    return anomaly_scores

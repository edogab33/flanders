from cgi import test
import matplotlib.pyplot as plt
import numpy as np
import os

def evaluate(
    threshold=0.005,
    alpha=1,
    gap_time=1,
    valid_start_point=0,
    valid_end_point=0,
    test_start_point=0,
    test_end_point=0,
    matrix_data_path='strategy/mscred/matrix_data/'):

    thred_b = threshold
    valid_start = int(valid_start_point/gap_time)
    valid_end = int(valid_end_point/gap_time)
    test_start = int(test_start_point/gap_time)
    test_end = int(test_end_point/gap_time)

    valid_anomaly_score = np.zeros((valid_end - valid_start , 1))
    test_anomaly_score = np.zeros((test_end - test_start, 1))

    test_data_path = matrix_data_path + "test_data/"
    reconstructed_data_path = matrix_data_path + "reconstructed_data/"

    for i in range(valid_start, test_end):
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

        if i < valid_end:
            valid_anomaly_score[i - valid_start] = num_broken
        else:
            test_anomaly_score[i - test_start] = num_broken
    if len(valid_anomaly_score) > 0:
        valid_anomaly_max = np.max(valid_anomaly_score.ravel())
    else:
        valid_anomaly_max = 0
    test_anomaly_score = test_anomaly_score.ravel()
    print(len(test_anomaly_score))

    #fig, axes = plt.subplots()
    #test_num = test_end - test_start
    #plt.ylim((min(test_anomaly_score), max(test_anomaly_score)+10))
    #plt.yticks(np.arange(min(test_anomaly_score), max(test_anomaly_score)+1, 10), fontsize = 10)
    #plt.plot(test_anomaly_score, 'b', linewidth = 2)
    #threshold = np.full((test_num), valid_anomaly_max * alpha)
    #axes.plot(threshold, color = 'black', linestyle = '--',linewidth = 2)
    #val = test_start
    #labels = []
    #for i in range(0, len(test_anomaly_score), gap_time):
    #    labels.append(str(int(val+i)))
    #plt.xticks(np.arange(0, len(test_anomaly_score), gap_time), fontsize = 10)
    #axes.set_xticklabels(labels, rotation = 25, fontsize = 10)
    #plt.xlabel('Test Time', fontsize = 25)
    #plt.ylabel('Anomaly Score', fontsize = 25)
    #axes.spines['right'].set_visible(False)
    #axes.spines['top'].set_visible(False)
    #axes.yaxis.set_ticks_position('left')
    #axes.xaxis.set_ticks_position('bottom')
    #fig.subplots_adjust(bottom=0.25)
    #fig.subplots_adjust(left=0.25)
    #plt.title("MSCRED", size = 25)
    #plt.show()
    threshold = 0.5
    print(threshold)
    for score in test_anomaly_score:
        print(score/(200*200))
        if score/(200*200) >= threshold:
            return True
    return False






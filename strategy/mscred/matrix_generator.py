import os
import numpy as np
import tensorflow as tf
import math

from .cnn import *

def generate_train_test_data(
    params_time_series=None,
    matrix_data_path="strategy/mscred/matrix_data/",
    train_start=0,
    train_end=0,
    test_start=0,
    test_end=0,
    gap_time=1,
    step_max=5,
    win_size=[1, 10, 20]):

    data = params_time_series.copy()
    print(data.shape)
    sensor_n = data.shape[0]

    # min-max normalization
    max_value = np.max(data, axis=1)
    min_value = np.min(data, axis=1)
    data = (np.transpose(data) - min_value)/(max_value - min_value + 1e-6)
    print(data.shape)

    data = np.transpose(data)
    print(data.shape)

    #multi-scale signature matix generation
    for w in range(len(win_size)):
        matrix_all = []
        win = win_size[w]
        print ("generating signature with window " + str(win) + "...")
        for t in range(test_start, test_end, gap_time):
            matrix_t = np.zeros((sensor_n, sensor_n))
            if t >= 60:
                for i in range(sensor_n):
                    for j in range(i, sensor_n):
                        matrix_t[i][j] = np.inner(data[i, t - win:t], data[j, t - win:t])/(win) # rescale by win
                        matrix_t[j][i] = matrix_t[i][j]
            matrix_all.append(matrix_t)

        path_temp = matrix_data_path + "matrix_win_" + str(win)

        # Check if the path exists, if not, create it:
        if not os.path.exists(matrix_data_path):
            os.makedirs(matrix_data_path)
        np.save(path_temp, matrix_all)
        del matrix_all[:]

    print("matrix generation finish!")

    #data sample generation
    print("generating train/test data samples...")

    value_colnames = ['total_count','error_count','error_rate']
    train_data_path = matrix_data_path + "train_data/"
    if not os.path.exists(train_data_path):
        os.makedirs(train_data_path)
    test_data_path = matrix_data_path + "test_data/"
    if not os.path.exists(test_data_path):
        os.makedirs(test_data_path)

    # Unify all signature matrices with window [10/30/60]
    data_all = []
    for value_col in value_colnames:
        for w in range(len(win_size)):
            path_temp = matrix_data_path + "matrix_win_" + str(win_size[w]) + ".npy"
            data_all.append(np.load(path_temp))

    train_test_time = [[train_start, train_end], [test_start, test_end]]
    for i in range(len(train_test_time)):
        for data_id in range(int(train_test_time[i][0]/gap_time), int(train_test_time[i][1]/gap_time)):
            step_multi_matrix = []
            for step_id in range(step_max, 0, -1):
                multi_matrix = []
                for k in range(len(value_colnames)):
                    for i in range(len(win_size)):
                        multi_matrix.append(data_all[k*len(win_size) + i][data_id - step_id])
                step_multi_matrix.append(multi_matrix)

            # Discriminate train and test data:
            if data_id >= (train_start/gap_time + win_size[-1]/gap_time + step_max) and data_id < (train_end/gap_time): # remove start points with invalid value
                path_temp = os.path.join(train_data_path, 'train_data_' + str(data_id))
                np.save(path_temp, step_multi_matrix)
            elif data_id >= (test_start/gap_time) and data_id < (test_end/gap_time):
                path_temp = os.path.join(test_data_path, 'test_data_' + str(data_id))
                np.save(path_temp, step_multi_matrix)

            del step_multi_matrix[:]

    print ("train/test data generation finish!")

def generate_reconstructed_matrices(
    model_path="strategy/mscred/model_ckpt/",
    test_data_path="strategy/mscred/matrix_data/test_data/",
    matrix_data_path = "strategy/mscred/matrix_data/",
    restore_idx=6,
    test_start_id=0,
    test_end_id=0,
    step_max=5,
    sensor_n=0,
    scale_n=9,
):
    data_input = tf.placeholder(tf.float32, [step_max, sensor_n, sensor_n, scale_n])

    # parameters: adding bias weight get similar performance
    conv1_W = tf.Variable(tf.zeros([3, 3, scale_n, 32]), name = "conv1_W")
    conv1_W = tf.get_variable("conv1_W", shape = [3, 3, scale_n, 32], initializer=tf.contrib.layers.xavier_initializer())
    conv2_W = tf.Variable(tf.zeros([3, 3, 32, 64]), name = "conv2_W")
    conv2_W = tf.get_variable("conv2_W", shape = [3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
    conv3_W = tf.Variable(tf.zeros([2, 2, 64, 128]), name = "conv3_W")
    conv3_W = tf.get_variable("conv3_W", shape = [2, 2, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
    conv4_W = tf.Variable(tf.zeros([2, 2, 128, 256]), name = "conv4_W")
    conv4_W = tf.get_variable("conv4_W", shape = [2, 2, 128, 256], initializer=tf.contrib.layers.xavier_initializer())
    
    conv1_out, conv2_out, conv3_out, conv4_out = cnn_encoder(data_input, conv1_W, conv2_W, conv3_W, conv4_W)
    conv1_out = tf.reshape(conv1_out, [-1, step_max, sensor_n, sensor_n, 32])
    conv2_out = tf.reshape(conv2_out, [-1, step_max, int(math.ceil(float(sensor_n)/2)), int(math.ceil(float(sensor_n)/2)), 64])
    conv3_out = tf.reshape(conv3_out, [-1, step_max, int(math.ceil(float(sensor_n)/4)), int(math.ceil(float(sensor_n)/4)), 128])
    conv4_out = tf.reshape(conv4_out, [-1, step_max, int(math.ceil(float(sensor_n)/8)), int(math.ceil(float(sensor_n)/8)), 256])

    conv1_lstm_attention_out, _, _ = conv1_lstm(conv1_out, sensor_n=sensor_n, step_max=step_max)
    conv2_lstm_attention_out, _, _ = conv2_lstm(conv2_out, sensor_n=sensor_n, step_max=step_max)
    conv3_lstm_attention_out, _, _ = conv3_lstm(conv3_out, sensor_n=sensor_n, step_max=step_max)
    conv4_lstm_attention_out, _, _ = conv4_lstm(conv4_out, sensor_n=sensor_n, step_max=step_max)

    deconv_out = cnn_decoder(conv1_lstm_attention_out, conv2_lstm_attention_out, conv3_lstm_attention_out, conv4_lstm_attention_out, sensor_n=sensor_n)
    saver = tf.train.Saver(max_to_keep = 10)

    with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=20, intra_op_parallelism_threads=20)) as sess:
        saver.restore(sess, model_path + str(restore_idx) + ".ckpt")
        reconstructed_data_path = matrix_data_path + "reconstructed_data/"
        if not os.path.exists(reconstructed_data_path):
            os.makedirs(reconstructed_data_path)
        print ("model test: generate recontrucuted matrices"+ "...")

        for test_id in range(test_start_id, test_end_id):
            matrix_data_path = test_data_path + 'test_data_' + str(test_id) + ".npy"
            matrix_gt = np.load(matrix_data_path)
            matrix_gt = np.transpose(matrix_gt, (0, 2, 3, 1))
            feed_dict = {data_input: np.asarray(matrix_gt)}
            reconstructed_matrix = sess.run([deconv_out], feed_dict)
            
            path_temp = os.path.join(reconstructed_data_path, 'reconstructed_data_' + str(test_id) + ".npy")
            np.save(path_temp, np.asarray(reconstructed_matrix))
        
        print ("reconstructed matrices generation finish.")
    tf.reset_default_graph()
import math
import tensorflow as tf

def cnn_encoder(input_matrix, conv1_W, conv2_W, conv3_W, conv4_W):
    '''
    Code adapted from: https://github.com/7fantasysz/MSCRED
    '''
    conv1 = tf.nn.conv2d(
      input = input_matrix,
      filter = conv1_W,
      strides=(1, 1, 1, 1),
      padding = "SAME")
    conv1 = tf.nn.selu(conv1)

    conv2 = tf.nn.conv2d(
      input = conv1,
      filter = conv2_W,
      strides=(1, 2, 2, 1),
      padding = "SAME")
    conv2 = tf.nn.selu(conv2)

    conv3 = tf.nn.conv2d(
      input = conv2,
      filter = conv3_W,
      strides=(1, 2, 2, 1),
      padding = "SAME")
    conv3 = tf.nn.selu(conv3)

    conv4 = tf.nn.conv2d(
      input = conv3,
      filter = conv4_W,
      strides=(1, 2, 2, 1),
      padding = "SAME")
    conv4 = tf.nn.selu(conv4)

    return  conv1, conv2, conv3, conv4

def cnn_decoder(conv1_lstm_out, conv2_lstm_out, conv3_lstm_out, conv4_lstm_out, sensor_n=0, scale_n=9):
    '''
    Code adapted from: https://github.com/7fantasysz/MSCRED
    '''
    conv1_lstm_out = tf.reshape(conv1_lstm_out, [1, sensor_n, sensor_n, 32])
    conv2_lstm_out = tf.reshape(conv2_lstm_out, [1, int(math.ceil(float(sensor_n)/2)), int(math.ceil(float(sensor_n)/2)), 64])
    conv3_lstm_out = tf.reshape(conv3_lstm_out, [1, int(math.ceil(float(sensor_n)/4)), int(math.ceil(float(sensor_n)/4)), 128])
    conv4_lstm_out = tf.reshape(conv4_lstm_out, [1, int(math.ceil(float(sensor_n)/8)), int(math.ceil(float(sensor_n)/8)), 256])

    deconv4_W = tf.Variable(tf.zeros([2, 2, 128, 256]), name = "deconv4_W")
    deconv4_W = tf.get_variable("deconv4_W", shape = [2, 2, 128, 256], initializer=tf.contrib.layers.xavier_initializer())
    deconv3_W = tf.Variable(tf.zeros([2, 2, 64, 256]), name = "deconv3_W")
    deconv3_W = tf.get_variable("deconv3_W", shape = [2, 2, 64, 256], initializer=tf.contrib.layers.xavier_initializer())
    deconv2_W = tf.Variable(tf.zeros([3, 3, 32, 128]), name = "deconv2_W")
    deconv2_W = tf.get_variable("deconv2_W", shape = [3, 3, 32, 128], initializer=tf.contrib.layers.xavier_initializer())
    deconv1_W = tf.Variable(tf.zeros([3, 3, scale_n, 64]), name = "deconv1_W")
    deconv1_W = tf.get_variable("deconv1_W", shape = [3, 3, scale_n, 64], initializer=tf.contrib.layers.xavier_initializer())

    deconv4 = tf.nn.conv2d_transpose(
      value = conv4_lstm_out,
      filter = deconv4_W,
      output_shape = [1, int(math.ceil(float(sensor_n)/4)), int(math.ceil(float(sensor_n)/4)), 128],
      strides = (1, 2, 2, 1),
      padding = "SAME")
    deconv4 = tf.nn.selu(deconv4)
    deconv4_concat = tf.concat([deconv4, conv3_lstm_out], axis = 3)

    deconv3 = tf.nn.conv2d_transpose(
      value = deconv4_concat,
      filter = deconv3_W,
      output_shape = [1, int(math.ceil(float(sensor_n)/2)), int(math.ceil(float(sensor_n)/2)), 64],
      strides = (1, 2, 2, 1),
      padding = "SAME")
    deconv3 = tf.nn.selu(deconv3)
    deconv3_concat = tf.concat([deconv3, conv2_lstm_out], axis = 3)

    deconv2 = tf.nn.conv2d_transpose(
      value = deconv3_concat,
      filter = deconv2_W,
      output_shape = [1, sensor_n, sensor_n, 32],
      strides = (1, 2, 2, 1),
      padding = "SAME")
    deconv2 = tf.nn.selu(deconv2)

    deconv2_concat = tf.concat([deconv2, conv1_lstm_out], axis = 3)

    deconv1 = tf.nn.conv2d_transpose(
      value = deconv2_concat,
      filter = deconv1_W,
      output_shape = [1, sensor_n, sensor_n, scale_n],
      strides = (1, 1, 1, 1),
      padding = "SAME")
    deconv1 = tf.nn.selu(deconv1)
    deconv1 = tf.reshape(deconv1, [1, sensor_n, sensor_n, scale_n])
    return deconv1

def conv1_lstm(conv1_out, sensor_n=0, step_max=5):
    '''
    Code adapted from: https://github.com/7fantasysz/MSCRED
    '''
    convlstm_layer = tf.contrib.rnn.ConvLSTMCell(
                conv_ndims = 2,
                input_shape = [sensor_n, sensor_n, 32],
                output_channels = 32,
                kernel_shape = [2, 2],
                use_bias = True,
                skip_connection = False,
                forget_bias = 1.0,
                initializers = None,
                name="conv1_lstm_cell")

    outputs, state = tf.nn.dynamic_rnn(convlstm_layer, conv1_out, time_major = False, dtype = conv1_out.dtype)

    # attention based on inner-product between feature representation of last step and other steps
    attention_w = []
    for k in range(step_max):
        attention_w.append(tf.reduce_sum(tf.multiply(outputs[0][k], outputs[0][-1]))/step_max)
    attention_w = tf.reshape(tf.nn.softmax(tf.stack(attention_w)), [1, step_max])

    outputs = tf.reshape(outputs[0], [step_max, -1])
    outputs = tf.matmul(attention_w, outputs)
    outputs = tf.reshape(outputs, [1, sensor_n, sensor_n, 32])

    return outputs, state[0], attention_w


def conv2_lstm(conv2_out, sensor_n=0, step_max=5):
    '''
    Code adapted from: https://github.com/7fantasysz/MSCRED
    '''
    convlstm_layer = tf.contrib.rnn.ConvLSTMCell(
                conv_ndims = 2,
                input_shape = [int(math.ceil(float(sensor_n)/2)), int(math.ceil(float(sensor_n)/2)), 64],
                output_channels = 64,
                kernel_shape = [2, 2],
                use_bias = True,
                skip_connection = False,
                forget_bias = 1.0,
                initializers = None,
                name="conv2_lstm_cell")

    outputs, state = tf.nn.dynamic_rnn(convlstm_layer, conv2_out, time_major = False, dtype = conv2_out.dtype)

    # attention based on inner-product between feature representation of last step and other steps
    attention_w = []
    for k in range(step_max):
        attention_w.append(tf.reduce_sum(tf.multiply(outputs[0][k], outputs[0][-1]))/step_max)
    attention_w = tf.reshape(tf.nn.softmax(tf.stack(attention_w)), [1, step_max])

    outputs = tf.reshape(outputs[0], [step_max, -1])
    outputs = tf.matmul(attention_w, outputs)
    outputs = tf.reshape(outputs, [1, int(math.ceil(float(sensor_n)/2)), int(math.ceil(float(sensor_n)/2)), 64])

    return outputs, state[0], attention_w


def conv3_lstm(conv3_out, sensor_n=0, step_max=5):
    '''
    Code adapted from: https://github.com/7fantasysz/MSCRED
    '''
    convlstm_layer = tf.contrib.rnn.ConvLSTMCell(
                conv_ndims = 2,
                input_shape = [int(math.ceil(float(sensor_n)/4)), int(math.ceil(float(sensor_n)/4)), 128],
                output_channels = 128,
                kernel_shape = [2, 2],
                use_bias = True,
                skip_connection = False,
                forget_bias = 1.0,
                initializers = None,
                name="conv3_lstm_cell")

    outputs, state = tf.nn.dynamic_rnn(convlstm_layer, conv3_out, time_major = False, dtype = conv3_out.dtype)

    attention_w = []
    for k in range(step_max):
        attention_w.append(tf.reduce_sum(tf.multiply(outputs[0][k], outputs[0][-1]))/step_max)
    attention_w = tf.reshape(tf.nn.softmax(tf.stack(attention_w)), [1, step_max])

    outputs = tf.reshape(outputs[0], [step_max, -1])
    outputs = tf.matmul(attention_w, outputs)
    outputs = tf.reshape(outputs, [1, int(math.ceil(float(sensor_n)/4)), int(math.ceil(float(sensor_n)/4)), 128])

    return outputs, state[0], attention_w


def conv4_lstm(conv4_out, sensor_n=0, step_max=5):
    '''
    Code adapted from: https://github.com/7fantasysz/MSCRED
    '''
    convlstm_layer = tf.contrib.rnn.ConvLSTMCell(
                conv_ndims = 2,
                input_shape = [int(math.ceil(float(sensor_n)/8)), int(math.ceil(float(sensor_n)/8)), 256],
                output_channels = 256,
                kernel_shape = [2, 2],
                use_bias = True,
                skip_connection = False,
                forget_bias = 1.0,
                initializers = None,
                name="conv4_lstm_cell")

    #initial_state = convlstm_layer.zero_state(batch_size, dtype = tf.float32)
    outputs, state = tf.nn.dynamic_rnn(convlstm_layer, conv4_out, time_major = False, dtype = conv4_out.dtype)

    attention_w = []
    for k in range(step_max):
        attention_w.append(tf.reduce_sum(tf.multiply(outputs[0][k], outputs[0][-1]))/step_max)
    attention_w = tf.reshape(tf.nn.softmax(tf.stack(attention_w)), [1, step_max])

    outputs = tf.reshape(outputs[0], [step_max, -1])
    outputs = tf.matmul(attention_w, outputs)
    outputs = tf.reshape(outputs, [1, int(math.ceil(float(sensor_n)/8)), int(math.ceil(float(sensor_n)/8)), 256])

    return outputs, state[0], attention_w
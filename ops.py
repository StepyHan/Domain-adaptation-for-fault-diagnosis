import tensorflow as tf



def conv2d(input_, output_dim, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02, name="conv2d", trainable=True):

    print('layer: %s, input: %s, output_dim: %s' % (name, input_, output_dim))
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev),
                            trainable=trainable)
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0),
                                 trainable=trainable)
        # conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
    return tf.nn.bias_add(conv, biases)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, regularizer=None, trainable=True):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev),
                                 trainable=trainable)

        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start),
                               trainable=trainable)
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(matrix))

        return tf.matmul(input_, matrix) + bias

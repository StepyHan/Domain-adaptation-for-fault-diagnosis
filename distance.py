import tensorflow as tf

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    :param source: [batch, -1]
    :param target: [batch, -1]
    :param kernel_mul:
    :param kernel_num: kernel numbers
    :param fix_sigma: sigma values of rbf
    :return: kernel matrix
    """

    source_batch = source.get_shape().as_list()[0]
    target_batch = target.get_shape().as_list()[0]

    n_samples = source_batch + target_batch

    total = tf.concat([source, target], axis=0)

    total0 = tf.tile(tf.expand_dims(total, 0), [n_samples, 1, 1])
    total1 = tf.tile(tf.expand_dims(total, 1), [1, n_samples, 1])

    L2_distance = tf.reduce_sum((total0-total1)**2, axis=2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = tf.reduce_sum(L2_distance) / (n_samples**2-n_samples)
    #以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

    kernel_val = [tf.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    #得到最终的核矩阵
    return sum(kernel_val) / len(kernel_val)



def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=1, fix_sigma=1, JDA_L=None):
    """
    :param source: source: [batch, feature size, feature size, channels]
    :param target: target: [batch, feature size, feature size, channels]
    :param kernel_mul:
    :param kernel_num: kernel numbers
    :param fix_sigma:
    :return: MMD Loss
    """
    # source = tf.Variable(tf.random_normal(shape=[2, 10], mean=0, stddev=1), name='v1')
    # target = tf.Variable(tf.random_normal(shape=[2, 10], mean=0, stddev=1), name='v1')
    source_batch = source.get_shape().as_list()[0]
    target_batch = target.get_shape().as_list()[0]
    source = tf.reshape(source, [source_batch, -1])
    target = tf.reshape(target, [target_batch, -1])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    XX = kernels[:source_batch, :source_batch]
    YY = kernels[source_batch:, source_batch:]
    XY = kernels[:source_batch, source_batch:]
    YX = kernels[source_batch:, :source_batch]

    if JDA_L == None:
        loss = tf.reduce_mean(XX + YY - XY - YX)
    else:
        loss = tf.trace(tf.matmul(kernels, JDA_L))
    return loss

def JDA_L(source_label, target_label):
    """
    :param source_label: [batch, dims] logit would be okay!
    :param target_label: [batch, dims] logit would be okay!
    :return: JDA L matrix
    """
    # source_label = tf.Variable([[10, 20], [12, 0], [1, 32]])
    # target_label = tf.Variable([[1, 0], [0, 1], [1, 0]])

    source_batch = source_label.get_shape().as_list()[0]
    target_batch = source_label.get_shape().as_list()[1]
    n_samples = source_batch + target_batch

    dims = source_label.get_shape().as_list()[1]

    source_label = tf.argmax(source_label, axis=1)
    target_label = tf.argmax(target_label, axis=1)

    L_list = []
    for c in range(dims):
        ys = tf.cast(tf.equal(source_label, c), tf.int32)
        yt = tf.cast(tf.equal(target_label, c), tf.int32)

        print('ys', ys)
        print('yt', yt)

        ns = tf.reduce_sum(ys)
        nt = tf.reduce_sum(yt)
        print("Fixxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxed")
        xx = tf.matmul(tf.reshape(ys, [-1, 1]), tf.reshape(ys, [1, -1])) / (ns ** 2 + 1)
        yy = tf.matmul(tf.reshape(yt, [-1, 1]), tf.reshape(yt, [1, -1])) / (nt ** 2 + 1)
        xy = tf.matmul(tf.reshape(ys, [-1, 1]), tf.reshape(yt, [1, -1])) / -(ns * nt + 1)
        yx = tf.matmul(tf.reshape(yt, [-1, 1]), tf.reshape(ys, [1, -1])) / -(ns * nt + 1)



        # concat together
        up = tf.concat([xx, xy], 1)
        bottom = tf.concat([yx, yy], 1)
        L = tf.concat([up, bottom], 0)
        L_list.append(L)

    return tf.cast(sum(L_list), tf.float32)

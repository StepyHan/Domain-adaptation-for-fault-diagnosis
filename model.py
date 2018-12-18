import os, imageio
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
import distance
from utils.wavelet_img import *
from ops import *

slim = tf.contrib.slim

class JDACNN():
    def __init__(self, checkpoint_dir, pretrain_lr, learning_rate,
                 source_data_path='./data/Wavelet_img_0HP',
                 target_data_path='./data/Wavelet_img_1HP',
                 img_height=64, img_width=64, img_channel=1,
                 batch_size=64, jdacnn_batch_size=100,
                 num_classes=4, cnn='LeNet', use_jda=True,
                 source_name='0HP',
                 target_name='1HP',
                 epoch=150,
                 gamma=1
                 ):
        self.batch_size = batch_size
        self.jdacnn_batch_size = jdacnn_batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.img_channel = img_channel
        self.num_classes = num_classes

        self.gamma = float(gamma)

        self.source_name = source_name
        self.target_name = target_name

        self.pretrain_lr = pretrain_lr
        self.learning_rate = learning_rate
        self.epoch = epoch

        self.checkpoint_dir = checkpoint_dir

        self.p_wavelet_imgs = Wav_img(path=source_data_path, normalization=False)
        self.s_wavelet_imgs = Wav_img(path=source_data_path, normalization=False)
        self.t_wavelet_imgs = Wav_img(path=target_data_path, normalization=False)

        self.cnn = cnn
        self.use_jda = use_jda

        self.pretrain_inputs = tf.placeholder(tf.float32, [self.batch_size,
                                                         self.img_height,
                                                         self.img_width,
                                                         self.img_channel], name='pretrain_inputs')

        self.pretrain_label = tf.placeholder(tf.float32, [self.batch_size,
                                                          self.num_classes], name='pretrain_label')

        self.source_inputs = tf.placeholder(tf.float32, [self.jdacnn_batch_size,
                                                         self.img_height,
                                                         self.img_width,
                                                         self.img_channel], name='source_inputs')

        self.target_inputs = tf.placeholder(tf.float32, [self.jdacnn_batch_size,
                                                         self.img_height,
                                                         self.img_width,
                                                         self.img_channel], name='target_inputs')

        self.source_label = tf.placeholder(tf.float32, [self.jdacnn_batch_size,
                                                        self.num_classes], name='source_label')

        self.target_label = tf.placeholder(tf.float32, [self.jdacnn_batch_size,
                                                        self.num_classes], name='target_label')

        # self.source_test_input = tf.placeholder(tf.float32, [None,
        #                                                      self.img_height,
        #                                                      self.img_width,
        #                                                      self.img_channel], name='source_test_inputs')
        #
        # self.source_test_label = tf.placeholder(tf.float32, [None,
        #                                                      self.img_height,
        #                                                      self.img_width,
        #                                                      self.img_channel], name='source_test_label')
        #
        # self.target_test_input = tf.placeholder(tf.float32, [None,
        #                                                      self.img_height,
        #                                                      self.img_width,
        #                                                      self.img_channel], name='target_test_inputs')
        #
        # self.target_test_label = tf.placeholder(tf.float32, [None,
        #                                                      self.img_height,
        #                                                      self.img_width,
        #                                                      self.img_channel], name='target_test_label')

    def pretrain(self):
        if self.cnn == 'LeNet':
            pretrain_logit, _ = self.LeNet(self.pretrain_inputs)
        elif self.cnn == 'Inception':
            pretrain_logit, _ = self.Inception(self.pretrain_inputs)
        else:
            raise Exception

        # get pretrain op
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pretrain_logit,
                                                                       labels=tf.argmax(self.pretrain_label, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        pretrain_loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
        pretrain_loss_sum = tf.summary.scalar("pretrain_loss", pretrain_loss)

        pretrain_optim = tf.train.AdamOptimizer(self.pretrain_lr).minimize(pretrain_loss)


        # get pretrain test accuracy
        test_data = self.p_wavelet_imgs.test_data
        test_label = self.p_wavelet_imgs.test_label

        pretest_input = tf.placeholder(tf.float32, [test_data.shape[0],
                                                    self.img_height,
                                                    self.img_width,
                                                    self.img_channel], name='pretest_inputs')

        pretest_label = tf.placeholder(tf.float32, [test_label.shape[0],
                                                    self.num_classes], name='pretest_label')

        if self.cnn == 'LeNet':
            pretest_logit, _ = self.LeNet(pretest_input, reuse=True)
        elif self.cnn == 'Inception':
            pretest_logit, _ = self.Inception(pretest_input, reuse=True)
        else:
            raise Exception

        correct_prediction = tf.equal(tf.argmax(pretest_logit, 1), tf.argmax(pretest_label, 1))
        accuray = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuray_sum = tf.summary.scalar("pretrain_accuracy", accuray)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            writer = tf.summary.FileWriter("./pretrain_logs", sess.graph)
            sess.run(tf.global_variables_initializer())
            counter = 0

            while self.p_wavelet_imgs._epochs_completed < 1000:
                batch_images, batch_labels = self.p_wavelet_imgs.next_batch(self.batch_size)
                _, summary_str, p_loss = sess.run([pretrain_optim, pretrain_loss_sum, pretrain_loss],
                                                  feed_dict={self.pretrain_inputs: batch_images,
                                                             self.pretrain_label: batch_labels})
                writer.add_summary(summary_str, counter)
                print("epoch:{}, train step:{}, loss:{}".format(self.p_wavelet_imgs._epochs_completed,
                                                                counter,
                                                                p_loss))

                if counter % 10 == 0:
                    test_acc, summary_str = sess.run([accuray, accuray_sum],
                                                     feed_dict={pretest_input: test_data,
                                                                pretest_label: test_label})
                    writer.add_summary(summary_str, counter)
                    print("epoch:{}, train step:{}, accuracy:{:04f}".format(self.p_wavelet_imgs._epochs_completed,
                                                                            counter, test_acc))

                if np.mod(counter, 500) == 2:
                    if not os.path.exists(self.checkpoint_dir):
                        os.makedirs(self.checkpoint_dir)
                    if not os.path.exists(os.path.join(self.checkpoint_dir, self.cnn)):
                        os.makedirs(os.path.join(self.checkpoint_dir, self.cnn))

                    save_path = saver.save(sess,
                                           os.path.join(os.path.join(self.checkpoint_dir, self.cnn), self.cnn),
                                           counter)
                    print("Model saved in file:%s" % save_path)
                counter += 1


    def build_model(self):
        if self.cnn == 'LeNet':
            s_logit, s_each_layer = self.LeNet(self.source_inputs, trainable=False)
        elif self.cnn == 'Inception':
            s_logit, s_each_layer = self.Inception(self.source_inputs, trainable=False)
        else:
            raise Exception

        # classification loss
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=s_logit,
                                                                       labels=tf.argmax(self.source_label, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        clc_loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

        # mmd loss
        if self.cnn == 'LeNet':
            t_logit, t_each_layer = self.LeNet(self.target_inputs, reuse=True, trainable=False)
            s_adaptation_layers = s_each_layer[2:]
            t_adaptation_layers = t_each_layer[2:]
        elif self.cnn == 'Inception':
            t_logit, t_each_layer = self.Inception(self.target_inputs, reuse=True, trainable=False)
            s_adaptation_layers = s_each_layer[2:]
            t_adaptation_layers = t_each_layer[2:]
        else:
            raise Exception

        mmd_loss_list = []
        mmd_sum = []
        for i, s, t in zip(range(len(s_adaptation_layers)), s_adaptation_layers, t_adaptation_layers):
            mmd = distance.mmd_rbf(s, t)
            print(mmd)
            mmd_loss_list.append(mmd)
            mmd_sum.append(tf.summary.scalar("mmd_adaptation_layer_%s" % i, mmd))


            if self.use_jda:
                jda_l = distance.JDA_L(self.source_label, t_logit)
                jda_mmd = distance.mmd_rbf(s, t, JDA_L=jda_l)
                print(jda_mmd)
                mmd_loss_list.append(jda_mmd)
                mmd_sum.append(tf.summary.scalar("jda_mmd_adaptation_layer_%s" % i, jda_mmd))

        mmd_loss = self.gamma * sum(mmd_loss_list)
        total_loss = mmd_loss + clc_loss

        mmd_sum = tf.summary.merge(mmd_sum)
        clc_loss_sum = tf.summary.scalar("jdacnn_train_clc_loss", clc_loss)
        loss_sum = tf.summary.scalar("jdacnn_train_loss", total_loss)
        jdacnn_optim = tf.train.AdamOptimizer(self.learning_rate).minimize(total_loss)

        # accuracy
        s_test_data = self.s_wavelet_imgs.test_data
        s_test_label = self.s_wavelet_imgs.test_label
        t_test_data = self.t_wavelet_imgs.test_data
        t_test_label = self.t_wavelet_imgs.test_label

        target_test_input = tf.placeholder(tf.float32, [t_test_data.shape[0],
                                                        self.img_height,
                                                        self.img_width,
                                                        self.img_channel], name='target_test_input')

        target_test_label = tf.placeholder(tf.float32, [t_test_label.shape[0],
                                                        self.num_classes], name='target_test_label')

        if self.cnn == 'LeNet':
            t_test_logit, _ = self.LeNet(target_test_input, reuse=True)
        elif self.cnn == 'Inception':
            t_test_logit, _ = self.Inception(target_test_input, reuse=True)
        else:
            raise Exception

        correct_prediction = tf.equal(tf.argmax(t_test_logit, 1), tf.argmax(target_test_label, 1))
        accuray = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuray_sum = tf.summary.scalar("target_accuracy", accuray)

        # train
        variables = slim.get_variables_to_restore()
        # variables_to_restore = [v for v in variables if 'adaptation_layer' not in v.name]
        variables_to_restore = [v for v in variables]
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:

            # TODO debug nan
            # from tensorflow.python import debug as tfdbg
            # sess = tfdbg.LocalCLIDebugWrapperSession(sess)
            # sess.add_tensor_filter("has_inf_or_nan", tfdbg.has_inf_or_nan)
            if not os.path.exists('./log'):
                os.makedirs('./log')

            writer = tf.summary.FileWriter("./log/use_jda_%s_log_%s_to_%s" %
                                           (self.source_name, self.use_jda, self.target_name), sess.graph)
            sess.run(tf.global_variables_initializer())

            ckpt = tf.train.get_checkpoint_state('checkpoint\LeNet')
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            print("RESTORE FROM: ------------------------------------")
            print(os.path.join(self.checkpoint_dir, self.cnn, ckpt_name))
            saver.restore(sess, os.path.join(self.checkpoint_dir, self.cnn, ckpt_name))

            new_saver = tf.train.Saver()

            counter = 0

            while self.t_wavelet_imgs._epochs_completed < self.epoch:
                s_batch_images, s_batch_labels = self.s_wavelet_imgs.next_batch(self.jdacnn_batch_size)
                t_batch_images, t_batch_labels = self.t_wavelet_imgs.next_batch(self.jdacnn_batch_size)
                _, summary_str, jda_ls, mmd_ls, mmd_str, clc_str = sess.run([jdacnn_optim, loss_sum, total_loss, mmd_loss, mmd_sum, clc_loss_sum],
                                                    feed_dict={self.source_inputs: s_batch_images,
                                                               self.source_label: s_batch_labels,
                                                               self.target_inputs: t_batch_images})
                writer.add_summary(summary_str, counter)
                writer.add_summary(mmd_str, counter)
                writer.add_summary(clc_str, counter)
                print("epoch:{}, train step:{}, total loss:{}, mmd loss:{}".format(self.s_wavelet_imgs._epochs_completed,
                                                                                   counter,
                                                                                   jda_ls,
                                                                                   mmd_ls))


                if counter % 10 == 0:
                    test_acc, acc_str = sess.run([accuray, accuray_sum],
                                                                   feed_dict={target_test_input: t_test_data,
                                                                              target_test_label: t_test_label})
                    writer.add_summary(acc_str, counter)
                    print("epoch:{}, train step:{}, accuracy:{:04f}".format(self.s_wavelet_imgs._epochs_completed,
                                                                            counter, test_acc))

                if np.mod(counter, 500) == 2:
                    if not os.path.exists(self.checkpoint_dir):
                        os.makedirs(self.checkpoint_dir)
                    if not os.path.exists(os.path.join(self.checkpoint_dir, 'JDACNN-%s' % self.cnn)):
                        os.makedirs(os.path.join(self.checkpoint_dir, 'JDACNN-%s' % self.cnn))

                    save_path = new_saver.save(sess,
                                               os.path.join(os.path.join(self.checkpoint_dir, 'JDACNN-%s' % self.cnn),
                                                            'JDACNN-%s' % self.cnn),
                                               counter)
                    print("Model saved in file:%s" % save_path)
                counter += 1

    def get_each_layer_mmd(self):

        # TODO cannot figure out OOM, sad ):
        source = self.s_wavelet_imgs.data[: 100]
        target = self.t_wavelet_imgs.data[: 100]
        source_inputs = tf.placeholder(tf.float32, [100,
                                                    self.img_height,
                                                    self.img_width,
                                                    self.img_channel], name='source_inputs')
        target_inputs = tf.placeholder(tf.float32, [100,
                                                    self.img_height,
                                                    self.img_width,
                                                    self.img_channel], name='target_inputs')

        s_logit, s_each_layer = self.LeNet(source_inputs, reuse=False, trainable=False)
        t_logit, t_each_layer = self.LeNet(target_inputs, reuse=True, trainable=False)

        mmd_loss_list = []
        s_each_layer = s_each_layer[:1]
        t_each_layer = t_each_layer[:1]
        for i, s, t in zip(range(len(s_each_layer)), s_each_layer, t_each_layer):
            mmd = distance.mmd_rbf(s, t)
            print(mmd)
            mmd_loss_list.append(mmd)



        variables = slim.get_variables_to_restore()
        variables_to_restore = [v for v in variables]
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            ckpt = tf.train.get_checkpoint_state('checkpoint\LeNet')
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            print("RESTORE FROM: ------------------------------------")
            print(os.path.join(self.checkpoint_dir, self.cnn, ckpt_name))
            saver.restore(sess, os.path.join(self.checkpoint_dir, self.cnn, ckpt_name))

            mmd = sess.run(mmd_loss_list, feed_dict={source_inputs: source,
                                                        target_inputs: target})


            print(mmd)



    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.cnn, self.cnn)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def LeNet(self, inputs, reuse=False, trainable=True):
        with tf.variable_scope("LeNet") as scope:
            if reuse:
                scope.reuse_variables()
            conv1 = tf.nn.relu(conv2d(inputs, 32, name='conv1', trainable=trainable))
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            conv2 = tf.nn.relu(conv2d(pool1, 64, name='conv2', trainable=trainable))
            pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            flatten = tf.reshape(pool2, [pool2.get_shape().as_list()[0], -1])

            fc1 = tf.nn.relu(linear(flatten, 1024, 'adaptation_layer_fc1',
                                    regularizer=tf.contrib.layers.l2_regularizer(0.0001)))

            fc2 = tf.nn.relu(linear(fc1, 512, 'adaptation_layer_fc2',
                                    regularizer=tf.contrib.layers.l2_regularizer(0.0001)))

            logit = linear(fc2, 4, 'fc1')

            return logit, [conv1, conv2, fc1, fc2]

    def Inception(self, inputs, reuse=False, trainable=True):
        return None, None

    def Inception_module(self):
        pass
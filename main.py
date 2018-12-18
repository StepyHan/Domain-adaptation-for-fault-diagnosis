import os
import numpy as np

from model import JDACNN

import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_float("pretrain_lr", 0.0002, "Learning rate of for adam [0.0002]")

flags.DEFINE_float("learning_rate", 0.00001, "Learning rate of for adam [0.0002]")

flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("img_height", 64, "The size of image to use")
flags.DEFINE_integer("img_width", 64, "The size of image to use")
flags.DEFINE_integer("jdacnn_batch_size", 100, "The size of image to use")
flags.DEFINE_integer("epoch", 1000, "epoch")
flags.DEFINE_integer("num_classes", 4, "num_classes")
flags.DEFINE_integer("gamma", 1, "weight of mmd loss")



flags.DEFINE_string("source_data_path", './data/Wavelet_img_0HP', "s path")
flags.DEFINE_string("target_data_path", './data/Wavelet_img_1HP', "t path")

flags.DEFINE_string("source_name", '0HP', "s name")
flags.DEFINE_string("target_name", '1HP', "t name")


flags.DEFINE_string("cnn", "LeNet", "LeNet or Inception")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")

flags.DEFINE_boolean("use_jda", False, "use jda or not")

FLAGS = flags.FLAGS

def main(_):

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)


    jdacnn = JDACNN(learning_rate=FLAGS.learning_rate,
                    pretrain_lr=FLAGS.pretrain_lr,
                    jdacnn_batch_size=FLAGS.jdacnn_batch_size,
                    batch_size=FLAGS.batch_size,
                    checkpoint_dir=FLAGS.checkpoint_dir,
                    img_height=FLAGS.img_height,
                    img_width=FLAGS.img_width,
                    source_data_path=FLAGS.source_data_path,
                    target_data_path=FLAGS.target_data_path,
                    source_name=FLAGS.source_name,
                    target_name=FLAGS.target_name,
                    cnn=FLAGS.cnn,
                    num_classes=FLAGS.num_classes,
                    use_jda=FLAGS.use_jda,
                    epoch=FLAGS.epoch,
                    gamma=FLAGS.gamma)

    jdacnn.build_model()

if __name__ == '__main__':
    tf.app.run()


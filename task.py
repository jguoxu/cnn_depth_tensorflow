#encoding: utf-8

from datetime import datetime
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf
from dataset import DataSet
from dataset import output_predict
import model
import train_operation as op
import metrics

MAX_STEPS = 200#10000000
MAX_RANGE = 3#1000

LOG_DEVICE_PLACEMENT = False
BATCH_SIZE = 8
TRAIN_FILE = "train.csv"

# directory to store coarse and refine network model.
REFINE_DIR = "refine"

# train with refine network if true, otherwise train with coarse network.
REFINE_TRAIN = True

FINE_TUNE = True

def train():
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        dataset = DataSet(BATCH_SIZE)

        # Get batch of tensors.
        # images is X, depth is Y, invalid_depth is matrix of 0 and 1 
        # that indicates which pixel in Y has depth value.
        images, depths, invalid_depths = dataset.csv_inputs(TRAIN_FILE)

        # keep_conv is the drop out rate in conv layers
        keep_conv = tf.placeholder(tf.float32)

        # TODO(xuguo): what's keep hidden?
        keep_hidden = tf.placeholder(tf.float32)

        if REFINE_TRAIN:
            # When training with refined network, train with both coarse and 
            # refined together.
            print("refine train.")
            coarse = model.inference(images, keep_conv, trainable=False)
            logits = model.inference_refine(images, coarse, keep_conv, keep_hidden)
        else:
            # When training with coarse network, train with only coarse network.
            # (mpng) this isn't called at all
            print("coarse train.")
            logits = model.inference(images, keep_conv, keep_hidden)

        # define loss function:
        # logits: the final output after FC layer.
        # depth: the Y-hat.
        # invalid_depth: the pixels without depth value.
        loss = model.loss(logits, depths, invalid_depths)

        # define trainning function, adam optimization.
        train_op = op.train(loss, global_step, BATCH_SIZE)

        # initialize all variable
        init_op = tf.global_variables_initializer()#tf.initialize_all_variables()

        # Session
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=LOG_DEVICE_PLACEMENT))
        sess.run(init_op)

        # parameters
        # the parameters are use to store checkpoint, so training can resume when exception
        # happens.
        model_param_dict = {}
        print('pre-trained parameters -----------------------------------')
        for variable in tf.global_variables():#tf.all_variables():
            variable_name = variable.name
            print("store model variables: " + str(variable_name))
            model_param_dict[variable_name] = variable

        model_saver = tf.train.Saver(model_param_dict)

        model_ckpt = tf.train.get_checkpoint_state(REFINE_DIR)
        if model_ckpt and model_ckpt.model_checkpoint_path:
            model_saver.restore(sess, model_ckpt.model_checkpoint_path)
            print("Pretrained model restored.")

        # train with multi-thread.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # step is epoch.
        for step in range(MAX_STEPS):
            index = 0
#             _, loss_value, logits_val, images_val = sess.run([train_op, loss, logits, images], feed_dict={keep_conv: 0.8, keep_hidden: 0.5})
#             print("%s: %d[epoch]: train loss %f" % (datetime.now(), step, loss_value))

            for i in range(MAX_RANGE):
                _, loss_value, logits_val, images_val = sess.run([train_op, loss, logits, images], feed_dict={keep_conv: 0.8, keep_hidden: 0.5})
                if index % 10 == 0:
                    print("%s: %d[epoch]: %d[iteration]: train loss %f" % (datetime.now(), step, index, loss_value))
                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                if index % 500 == 0:
                    output_predict(logits_val, images_val, "data/predict_refine_%05d_%05d" % (step, i))
                index += 1

            # save parameters every 5 epoch.
            if step % 5000 == 0 or (step * 1) == MAX_STEPS:
                checkpoint_path = REFINE_DIR + '/model.ckpt'
                model_saver.save(sess, checkpoint_path, global_step=step)

        coord.request_stop()
        coord.join(threads)
        sess.close()


def main(argv=None):
    if not gfile.Exists(REFINE_DIR):
        gfile.MakeDirs(REFINE_DIR)
    train()


if __name__ == '__main__':
    tf.app.run()

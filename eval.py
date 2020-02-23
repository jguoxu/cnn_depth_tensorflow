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

BATCH_SIZE = 1
EVAL_FILE = "test.csv"

# directory to store coarse and refine network model.
COARSE_DIR = "coarse"
REFINE_DIR = "refine"

def eval():
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        dataset = DataSet(BATCH_SIZE)

        # Get batch of tensors.
        # images is X, depth is Y, invalid_depth is matrix of 0 and 1 
        # that indicates which pixel in Y has depth value.
        images, depths, invalid_depths = dataset.csv_inputs(EVAL_FILE)

        # keep_conv is the drop out rate in conv layers
        keep_conv = tf.placeholder(tf.float32)

        # TODO(xuguo): what's keep hidden?
        keep_hidden = tf.placeholder(tf.float32)

        coarse = model.inference(images, keep_conv, trainable=False)
        logits = model.inference_refine(images, coarse, keep_conv, keep_hidden)

        # initialize all variable
        init_op = tf.global_variables_initializer()

        # Session
        sess = tf.Session()
        sess.run(init_op)

        # load pre-trained parameters
        coarse_params = {}
        refine_params = {}

        for variable in tf.global_variables():
            variable_name = variable.name
            print("parameter: %s" % (variable_name))
            if variable_name.find("/") < 0 or variable_name.count("/") != 1:
                continue
            if variable_name.find('coarse') >= 0:
                coarse_params[variable_name] = variable
            print("parameter: %s" %(variable_name))
            if variable_name.find('fine') >= 0:
                refine_params[variable_name] = variable

        # define saver
        saver_coarse = tf.train.Saver(coarse_params)
        saver_refine = tf.train.Saver(refine_params)
        
        coarse_ckpt = tf.train.get_checkpoint_state(COARSE_DIR)
        if coarse_ckpt and coarse_ckpt.model_checkpoint_path:
            print("Pretrained coarse Model Loading.")
            saver_coarse.restore(sess, coarse_ckpt.model_checkpoint_path)
            print("Pretrained coarse Model Restored.")
        else:
            print("No Pretrained coarse Model.")
        
        
        refine_ckpt = tf.train.get_checkpoint_state(REFINE_DIR)
        if refine_ckpt and refine_ckpt.model_checkpoint_path:
            print("Pretrained refine Model Loading.")
            saver_refine.restore(sess, refine_ckpt.model_checkpoint_path)
            print("Pretrained refine Model Restored.")
        else:
            print("No Pretrained refine Model.")

        # train with multi-thread.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        logits_val, images_val = sess.run([logits, images], feed_dict={keep_conv: 1.0, keep_hidden: 0.5})

        print("logits shape:" + str(logits_val.shape))
        print("images_val shape:" + str(images_val.shape))
        print("depths shape :" + str(depths.shape))
        print("invalid_depths shape :" + str(invalid_depths.shape))

        metric_log_error_fn = metrics.log_error(logits_val, depths, invalid_depths)
        metric_log_error = sess.run([metric_log_error_fn])

        print ("metric_log_error: " + str(metric_log_error))

        coord.request_stop()
        coord.join(threads)
        sess.close()


def main(argv=None):
    assert gfile.Exists(COARSE_DIR), "evaluation requires coarse network weights"
    assert gfile.Exists(REFINE_DIR), "evaluation requires refine network weights"
    eval()


if __name__ == '__main__':
    tf.app.run()

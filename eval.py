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

RUN_EVAL_ON_TRAIN = True
TRAIN_FILE = "train.csv"
EVAL_FILE = "test.csv"
REFINE_DIR = "refine"
COARSE_DIR = "coarse"

def eval():
    eval_size = len(open(EVAL_FILE).readlines())
    print ('evaluating %d files' % eval_size)

    with tf.Graph().as_default():
        dataset = DataSet(eval_size)

        images_test, depths_test, invalid_depths_test = dataset.csv_inputs(EVAL_FILE)
        images_train, depths_train, invalid_depths_train = dataset.csv_inputs(TRAIN_FILE)

        # keep_conv is the drop out rate in conv layers
        keep_conv = tf.placeholder(tf.float32)
        keep_hidden = tf.placeholder(tf.float32)

        coarse_test = model.inference(images_test, trainable=False)
        logits_test = model.inference_refine(images_test, coarse_test, keep_conv, trainable=False)
        
        metric_log_error_test = metrics.log_error(logits_test, depths_test, invalid_depths_test)
        scale_invariant_error_test = metrics.scale_invariant_error(logits_test, depths_test, invalid_depths_test)
        
        coarse_train = model.inference(images_train, reuse=True, trainable=False)
        logits_train = model.inference_refine(images_train, coarse_train, keep_conv, reuse=True, trainable=False)
        loss_train = model.loss(logits_train, depths_train, invalid_depths_train)
        
        metric_log_error_train = metrics.log_error(logits_train, depths_train, invalid_depths_train)
        scale_invariant_error_train = metrics.scale_invariant_error(logits_train, depths_train, invalid_depths_train)

        # initialize all variable
        init_op = tf.global_variables_initializer()

        # Session
        sess = tf.Session()
        sess.run(init_op)

        # load pre-trained parameters
        refine_params = {}
        coarse_params = {}

        for variable in tf.global_variables():
            variable_name = variable.name
            print(variable_name)
            if variable_name.find("/") < 0 or variable_name.count("/") != 1:
                continue
            if variable_name.find('coarse') >= 0:
                coarse_params[variable_name] = variable
            if variable_name.find('fine') >= 0:
                refine_params[variable_name] = variable

        saver_refine = tf.train.Saver(refine_params)
        saver_coarse = tf.train.Saver(coarse_params)

        refine_ckpt = tf.train.get_checkpoint_state(REFINE_DIR)
        coarse_ckpt = tf.train.get_checkpoint_state(COARSE_DIR)
        if coarse_ckpt and coarse_ckpt.model_checkpoint_path:
            print("Pretrained coarse Model Loading.")
            saver_coarse.restore(sess, coarse_ckpt.model_checkpoint_path)
            print("Pretrained coarse Model Restored.")
        if refine_ckpt and refine_ckpt.model_checkpoint_path:
            print("Pretrained refine Model Loading.")
            saver_refine.restore(sess, refine_ckpt.model_checkpoint_path)
            print("Pretrained refine Model Restored.")
        else:
            print("No Pretrained refine Model.")

        # train with multi-thread.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        logits_val_test, images_val_test, metric_log_error_test, scale_invariant_error_test = sess.run([logits_test, images_test, metric_log_error_test, scale_invariant_error_test], feed_dict={keep_conv: 1.0, keep_hidden: 1.0})
        
        output_predict(logits_val_test, images_val_test, "data/predict_eval_test")
        
#         print("logits shape:" + str(logits_val_test.shape))
#         print("images_val shape:" + str(images_val_test.shape))
#         print("depths shape :" + str(depths_test.shape))
#         print("invalid_depths shape :" + str(invalid_depths_test.shape))

        print ("Eval metrics:")
        print ("metric_log_error: " + str(metric_log_error_test))
        print ("scale_invariant_error: " + str(scale_invariant_error_test))
        
        if RUN_EVAL_ON_TRAIN:
            logits_val_train, images_val_train, loss_train, metric_log_error_train, scale_invariant_error_train = sess.run([logits_train, images_train, loss_train, metric_log_error_train, scale_invariant_error_train], feed_dict={keep_conv: 1.0, keep_hidden: 1.0})
            output_predict(logits_val_train, images_val_train, "data/predict_eval_train")
            
            print ("Train eval metrics:")
            print ("Loss: " + str(loss_train))
            print ("metric_log_error: " + str(metric_log_error_train))
            print ("scale_invariant_error: " + str(scale_invariant_error_train))
        

        coord.request_stop()
        coord.join(threads)
        sess.close()


def main(argv=None):
    assert gfile.Exists(REFINE_DIR), "evaluation requires refine network weights"
    eval()


if __name__ == '__main__':
    tf.app.run()

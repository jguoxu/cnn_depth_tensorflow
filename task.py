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

TRAIN_STEPS = 2 #mpng: small epoch for exporation
TRAIN_RANGE = 2 #mpng: small range for exploration

#TODO (mpng): Put max steps and range back when ready
#MAX_STEPS = 10000000
MAX_STEPS = TRAIN_STEPS
#MAX_RANGE = 1000
MAX_RANGE = TRAIN_RANGE

LOG_DEVICE_PLACEMENT = True
BATCH_SIZE = 8
TRAIN_FILE = "train.csv"

# directory to store coarse and refine network model.
COARSE_DIR = "coarse"
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
        coarse_params = {}
        refine_params = {}
        if REFINE_TRAIN:
            for variable in tf.global_variables():#tf.all_variables():
                variable_name = variable.name
                print("parameter: %s" % (variable_name))
                if variable_name.find("/") < 0 or variable_name.count("/") != 1:
                    continue
                if variable_name.find('coarse') >= 0:
                    coarse_params[variable_name] = variable
                print("parameter: %s" %(variable_name))
                if variable_name.find('fine') >= 0:
                    refine_params[variable_name] = variable
        else:
            for variable in tf.trainable_variables():
                variable_name = variable.name
                print("parameter: %s" %(variable_name))
                if variable_name.find("/") < 0 or variable_name.count("/") != 1:
                    continue
                if variable_name.find('coarse') >= 0:
                    coarse_params[variable_name] = variable
                if variable_name.find('fine') >= 0:
                    refine_params[variable_name] = variable
        # define saver
        print(coarse_params)
        saver_coarse = tf.train.Saver(coarse_params)
        if REFINE_TRAIN:
            saver_refine = tf.train.Saver(refine_params)
        # fine tune
        if FINE_TUNE:
            coarse_ckpt = tf.train.get_checkpoint_state(COARSE_DIR)
            if coarse_ckpt and coarse_ckpt.model_checkpoint_path:
                print("Pretrained coarse Model Loading.")
                saver_coarse.restore(sess, coarse_ckpt.model_checkpoint_path)
                print("Pretrained coarse Model Restored.")
            else:
                print("No Pretrained coarse Model.")
            if REFINE_TRAIN:
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

        # step is epoch.
        for step in range(MAX_STEPS):
            index = 0
            #TODO(xuguo): is 'i' mini batches? where is the dividing of batches?
            for i in range(MAX_RANGE):
                # start trainning:
                # loss_value - loss
                # logits_val - Y-hat
                # image_val - X
                _, loss_value, logits_val, images_val = sess.run([train_op, loss, logits, images], feed_dict={keep_conv: 0.8, keep_hidden: 0.5})
                if index % 10 == 0:
                    print("%s: %d[epoch]: %d[iteration]: train loss %f" % (datetime.now(), step, index, loss_value))
                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                if index % 500 == 0:
                    if REFINE_TRAIN:
                        output_predict(logits_val, images_val, "data/predict_refine_%05d_%05d" % (step, i))
                    else:
                        output_predict(logits_val, images_val, "data/predict_%05d_%05d" % (step, i))

                #TODO (mpng): Output prediction at every step, i for testing. To remove. 
                output_predict(logits_val, images_val, "data/playground_%05d_%05d" %(step, i))        
                index += 1

            # save parameters every 5 epoch.
            if step % 5 == 0 or (step * 1) == MAX_STEPS:
                if REFINE_TRAIN:
                    refine_checkpoint_path = REFINE_DIR + '/model.ckpt'
                    saver_refine.save(sess, refine_checkpoint_path, global_step=step)
                else:
                    coarse_checkpoint_path = COARSE_DIR + '/model.ckpt'
                    saver_coarse.save(sess, coarse_checkpoint_path, global_step=step)

        #Calculate metrics when done with training
        print("Logits shape: ", logits_val.shape)
        print("Depths shape: ", images_val.shape)
        metric_log_error = metrics.log_error(logits_val, images_val, invalid_depths)
        print("Log Error: ", metric_log_error)

        metric_scale_invariant_error = metrics.scale_invariant_error(logits_val, images_val, invalid_depths)
        print("Scale Invariant Error: ", metric_scale_invariant_error)

        coord.request_stop()
        coord.join(threads)
        sess.close()


def main(argv=None):
    if not gfile.Exists(COARSE_DIR):
        gfile.MakeDirs(COARSE_DIR)
    if not gfile.Exists(REFINE_DIR):
        gfile.MakeDirs(REFINE_DIR)
    train()


if __name__ == '__main__':
    tf.app.run()

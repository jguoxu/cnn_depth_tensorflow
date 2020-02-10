# encoding: utf-8

import tensorflow as tf

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 500
NUM_EPOCHS_PER_DECAY = 30
INITIAL_LEARNING_RATE = 0.0001
LEARNING_RATE_DECAY_FACTOR = 0.9
MOVING_AVERAGE_DECAY = 0.999999


def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
    return loss_averages_op


def train(total_loss, global_step, batch_size):
    num_batches_per_epoch = float(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN) / batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # learning rate with discrete staircase decay.
    lr = tf.train.exponential_decay(
        INITIAL_LEARNING_RATE,
        global_step,
        decay_steps,
        LEARNING_RATE_DECAY_FACTOR,
        staircase=True)
    
    # tf summary is a debug function, so that learning rate shows up in a summary
    # dashboard.
    tf.summary.scalar('learning_rate', lr)
    # compute the loss function's exponential moving average.
    # and add it to dashboard.
    loss_averages_op = _add_loss_summaries(total_loss)

    # tf.control_dependencies make sure loss_averages_op is evaluate before the
    # scoped blocked.
    with tf.control_dependencies([loss_averages_op]):
        # forward prop
        opt = tf.train.AdamOptimizer(lr)
        # back prop
        grads = opt.compute_gradients(total_loss)
    # w = w - learning_rate * dW
    # b = b - learning_rate * dB
    # global_step increase by 1 after this operation, global_step is a out variable.
    # https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/Optimizer
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # add trainable varibale histogram to summary
    # trainable variales are: trainable=True, essentially the weights in model_parts.py
    for var in tf.trainable_variables():
        print(var.op.name)
        tf.summary.histogram(var.op.name, var)

    # add dW histogram to summary
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # apply moving average delay to weights?? why not to dW?
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op

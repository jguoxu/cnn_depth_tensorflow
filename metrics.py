import tensorflow as tf

# Computes mean of (square(logy - logy*))
def log_error(logits, depths, invalid_depths):
    logits_flat = tf.reshape(logits, [-1, 55*74])
    depths_flat = tf.reshape(depths, [-1, 55*74])
    invalid_depths_flat = tf.reshape(invalid_depths, [-1, 55*74])
    predict = tf.multiply(logits_flat, invalid_depths_flat)
    target = tf.multiply(depths_flat, invalid_depths_flat)
    d = tf.subtract(predict, target)
    square_d = tf.square(d)
    sum_square_d = tf.reduce_sum(square_d, 1)
    cost = tf.reduce_mean(sum_square_d / 55.0*74.0)
    tf.add_to_collection('error', cost)
    return tf.add_n(tf.get_collection('error'), name='total_error')
import tensorflow as tf
import numpy as np

def softmax(x, axis=1):
    if axis == 0:
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    elif axis == 1:
        return np.exp(x) / np.expand_dims(np.sum(np.exp(x), axis=1), axis=1)


def rand_weight_drop(weight, shape, keep_rate):
    nb_weight_elem = shape[0] * shape[1]
    nb_kp = round(nb_weight_elem * keep_rate)
    nb_dp = nb_weight_elem - nb_kp
    kp_ = np.ones([nb_kp, ])
    dp_ = np.zeros([nb_dp, ])
    kp_dp = np.concatenate([kp_, dp_])
    np.random.shuffle(kp_dp)
    kp_dp_reshape = np.reshape(kp_dp, weight.shape)
    weight_drop = tf.multiply(weight, kp_dp_reshape)

    return weight_drop


def pinv_func1(H, omega=1.):
    identity = tf.constant(np.identity(10000), dtype=tf.float32)  # the 10000 is the test batch_size
    H_T = tf.transpose(H)
    H_ = tf.matmul(H_T, tf.matrix_inverse(tf.matmul(H, H_T) + identity / omega))

    return H_


def pinv_func2(H, omega=1.):
    identity = tf.constant(np.identity(H.shape[1]), dtype=tf.float32)
    H_T = tf.transpose(H)
    H_ = tf.matmul(tf.matrix_inverse(tf.matmul(H_T, H) + identity / omega), H_T)

    return H_

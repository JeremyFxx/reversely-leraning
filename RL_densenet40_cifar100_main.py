import keras
import keras.backend as K
import tensorflow as tf
from keras.datasets import cifar100

from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

import time
import os
import argparse
import datetime
import socket

import numpy as np
import json
import pickle

from plot_results import plot_results
from densenet40 import densenet_model, preprocess_data
from reversely_learning_func import *

# Parse the commandline argument
parser = argparse.ArgumentParser(description='Reverse learning.')
parser.add_argument('-r', '--reverse',
                    action = 'store_true',
                    dest='reverse',
                    default=False,
                    help='Enable reversely learning.')
parser.add_argument('-g', '--igpu', type=int, default=0,
                    help='Index of gpu')
parser.add_argument('-t','--itest', type=str, default='default',
                    help='Short description of this experiment.')
args = parser.parse_args()

# GPU SELECT
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.igpu)

# TEST SETTING
TEST_SETTING = args.itest

# NET / DATASET
MODEL = 'densenet40'
DATASET = 'cifar100'

# RESULTS
RESULTS_BASE_DIR = './BTRL_%s_%s_%s/' % (MODEL, DATASET, TEST_SETTING)

WEIGHTS_DIR = RESULTS_BASE_DIR + 'weights_%s/' % (TEST_SETTING)
weights_file = WEIGHTS_DIR + '%s_%s_%s_newest.h5' % (MODEL, DATASET, TEST_SETTING)


HIST_DIR = RESULTS_BASE_DIR + 'hist_%s/' % (TEST_SETTING)
hist_pi_file = HIST_DIR + '%s_%s_%s_hist_pi' % (MODEL, DATASET, TEST_SETTING)
hist_json_file = HIST_DIR + '%s_%s_%s_hist.json' % (MODEL, DATASET, TEST_SETTING)

dir_list = [WEIGHTS_DIR, HIST_DIR]
for dir in dir_list:
    if not os.path.exists(dir):
        os.makedirs(dir)



def train_model():
    batch_size = 64
    nb_epoch = 300
    depth = 40
    nb_layers = 12
    nb_dense_block = 3
    nb_filter = 16
    growth = 12
    dropout_rate = 0.2
    learning_rate = 0.1
    weight_decay = 1e-4
    nb_classes = 100

    rl_up_rate = 0.1
    rl_epoch = 5            # number of interval epoches between 2 reversely learning session
    rl_loop_epoch = 1   # number of loops in each reversely learning session
    dim_fc1 = 448           # the dimension of the bottleneck layer of densenet-40


    ##  BP-based learning setting
    print('Building the model...\n')
    model = densenet_model(nb_classes, nb_dense_block, nb_layers, growth, dropout=dropout_rate)
    model.summary()

    sgd = SGD(lr=learning_rate, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    print('Model compiled.')

    ##  Reversely learning setting
    with tf.device('/gpu:0'):
        with tf.name_scope('config_para'):
            omega = 2e-1
            weights_elem_keep_rate = 0.5
            rl_ur_ph = tf.placeholder(tf.float32, name='rl_ur_ph')

        with tf.name_scope('forward_pass_rl'):
            y_ = tf.placeholder(tf.float32, [None, nb_classes], name='y_ph')

            w_fc12_ph = tf.placeholder(tf.float32, [dim_fc1, nb_classes], name='w_fc12_ph')

            fc1_ph = tf.placeholder(tf.float32, [None, dim_fc1], name='fc1_ph')
            fc2_ts = tf.nn.relu(tf.matmul(fc1_ph, w_fc12_ph), name='fc2_ts')

        with tf.name_scope('accuracy_forward_rl'):
            correct_prediction_forward_rl = tf.equal(tf.argmax(fc2_ts, 1), tf.argmax(y_, 1))
            correct_prediction_forward_rl = tf.cast(correct_prediction_forward_rl, tf.float32)
            accuracy_forward_rl = tf.reduce_mean(correct_prediction_forward_rl)

        with tf.name_scope('reverse_learning_rl'):
            # fc3 reverse learning
            w_fc12_rl_tmp_ts = tf.matmul(pinv_func2(fc1_ph), y_,
                                     name='w_fc12_rl_diff_ts')  # use fc1_ts and y_ to compute the w_fc12

            w_fc12_rl_dif_dp_ts = rand_weight_drop(w_fc12_rl_tmp_ts, shape=[dim_fc1, nb_classes],
                                                   keep_rate=weights_elem_keep_rate)
            w_fc12_rl_ts = tf.add(w_fc12_ph, w_fc12_rl_dif_dp_ts * rl_ur_ph, name='w_fc12_rl_ts')

    ## Loading data
    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train = preprocess_data(x_train)
    x_test = preprocess_data(x_test)

    y_train = keras.utils.to_categorical(y_train, nb_classes)
    y_test = keras.utils.to_categorical(y_test, nb_classes)
    print('Data loaded.\n\n')

    ## Callbacks and history
    model_checkpoint = ModelCheckpoint(weights_file, monitor="val_acc", save_best_only=True,
                                       save_weights_only=True, verbose=1)
    callbacks = [model_checkpoint]

    # hist holder

    #
    list_bp_train_loss = []
    list_bp_test_loss = []
    list_bp_train_acc = []
    list_bp_test_acc = []

    #
    list_w12_test_acc = []


    #
    hist_final = {}

    #########################
    #       Main loop       #
    #########################
    for epoch in range(1, nb_epoch + 1):
        #########################
        # learning rate control #
        #########################
        if (epoch - 1) == int(0.5 * nb_epoch):
            learning_rate = 0.01
            K.set_value(model.optimizer.lr, learning_rate)
            print('Epoch %d: Set learning rate to %s, and model is compiled' % (epoch, learning_rate))
            rl_up_rate = rl_up_rate / 10.

        if (epoch - 1) == int(0.75 * nb_epoch):
            learning_rate = 0.001
            K.set_value(model.optimizer.lr, learning_rate)
            print('Epoch %d: Set learning rate to %s, and model is compiled' % (epoch, learning_rate))
            rl_up_rate = rl_up_rate / 10.

        print('Epoch %d: BP learning rate: %s; Reversely learning update rate: %s. ' % (
        epoch, K.get_value(model.optimizer.lr), rl_up_rate))
        hist = model.fit(x_train, y_train, batch_size=batch_size,
                      epochs=epoch, verbose=2,
                      validation_data=(x_test, y_test),
                      callbacks=callbacks, shuffle=True, initial_epoch=epoch-1)

        # eval_loss, eval_acc = model.evaluate(x_test, y_test, verbose=0, batch_size=32)
        # print('Epoch %d: Evaluate acc before reverse learning: %s.\n ' % (epoch, eval_acc))

        ########################
        #     hist and plot    #
        ########################
        list_bp_train_loss.append(hist.history['loss'])
        list_bp_test_loss.append(hist.history['val_loss'])
        list_bp_train_acc.append(hist.history['acc'])
        list_bp_test_acc.append(hist.history['val_acc'])


        ############################
        #     REVERSE LEARNING     #
        ############################
        if ( epoch-1 ) % rl_epoch ==0:
            print('Epoch %d: Begin reversely learning...' % epoch)

            model_ft = Model(model.input, outputs=model.get_layer(name='fc1').output)
            fc1_tr_np = model_ft.predict(x_train)
            fc1_te_np = model_ft.predict(x_test)
            for idx_RLloop in range(rl_loop_epoch):
                # print('Epoch %d, idx_RLloop %d...' % (epoch, idx_RLloop))
                w_fc12_np = model.get_layer(name='fc2').get_weights()[0]

                ## Execute reversely learning
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    start_time = time.time()

                    w_fc12_rl_np = w_fc12_rl_ts.eval(
                        feed_dict={
                            fc1_ph: fc1_tr_np,
                            y_: y_train,
                            w_fc12_ph: w_fc12_np,
                            rl_ur_ph: rl_up_rate
                        })

                    acc_tf = accuracy_forward_rl.eval(
                        feed_dict={
                            fc1_ph: fc1_te_np,
                            y_: y_test,
                            w_fc12_ph: w_fc12_rl_np
                        })

                    print('Epoch {}, idx_RLloop {} Reversely learning: acc with updated weight: {}, time duration: {}.\n'.format(epoch, idx_RLloop, acc_tf, time.time() - start_time))
                    if idx_RLloop == (rl_loop_epoch - 1):
                        list_w12_test_acc.append([acc_tf.item()])

                #### update the reversely learnt weight to the model
                w_fc12_list = [None]
                w_fc12_list[0] = w_fc12_np
                model.get_layer(name='fc2').set_weights(w_fc12_list)

        # write the hist
        hist_final['loss'] = list_bp_train_loss
        hist_final['val_loss'] = list_bp_test_loss
        hist_final['acc'] = list_bp_train_acc
        hist_final['val_acc'] = list_bp_test_acc
        hist_final['w12_acc'] = list_w12_test_acc

        with open(hist_json_file, 'w') as file_json:
            json.dump(hist_final, file_json, indent=4, sort_keys=True)
        with open(hist_pi_file, 'wb') as file_pi:
            pickle.dump(hist_final, file_pi)

    # after the all the loop, plot the
    plot_results(hist_json_file, save_dir=HIST_DIR, mode='json')

    return

if __name__ == '__main__':
    train_model()

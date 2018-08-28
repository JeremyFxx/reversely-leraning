import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Activation, Dropout, Dense, Conv2D, concatenate
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import numpy as np

def add_layer(x, nb_channels, kernel_size=3, dropout=0., l2_reg=1e-4):
    out = BatchNormalization(gamma_regularizer=l2(l2_reg),
                             beta_regularizer=l2(l2_reg))(x)
    out = Activation('relu')(out)
    out = Conv2D(nb_channels, (kernel_size, kernel_size),
                        padding='same', kernel_initializer='he_normal',
                        kernel_regularizer=l2(l2_reg), use_bias=False)(out)
    if dropout > 0:
        out = Dropout(dropout)(out)
    return out

def dense_block(x, nb_layers, growth_rate, dropout=0., l2_reg=1e-4):
    for i in range(nb_layers):
        # Get layer output
        out = add_layer(x, growth_rate, dropout=dropout, l2_reg=l2_reg)
        if K.image_dim_ordering() == 'tf':
            merge_axis = -1
        elif K.image_dim_ordering() == 'th':
            merge_axis = 1
        else:
            raise Exception('Invalid dim_ordering: ' + K.image_dim_ordering())
        # Concatenate input with layer ouput
        x = concatenate([x, out], axis=merge_axis)
    return x

def transition_block(x, nb_channels, dropout=0., l2_reg=1e-4):
    x = add_layer(x, nb_channels, kernel_size=1, dropout=dropout, l2_reg=l2_reg)
    # x = Convolution2D(n_channels, 1, 1, border_mode='same',
    #                   init='he_normal', W_regularizer=l2(l2_reg))(x)
    x = AveragePooling2D()(x)
    return x

def densenet_model(nb_classes, nb_blocks, nb_layers, growth_rate, dropout=0., l2_reg=1e-4,
                   init_channels=16):
    n_channels = init_channels
    inputs = Input(shape=(32, 32, 3))
    x = Conv2D(init_channels, (3, 3), padding='same',
                      kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg),
               use_bias=False)(inputs)
    for i in range(nb_blocks - 1):
        # Create a dense block
        x = dense_block(x, nb_layers, growth_rate,
                        dropout=dropout, l2_reg=l2_reg)
        # Update the number of channels
        n_channels += nb_layers*growth_rate
        # Transition layer
        x = transition_block(x, n_channels, dropout=dropout, l2_reg=l2_reg)

    # Add last dense_block
    x = dense_block(x, nb_layers, growth_rate, dropout=dropout, l2_reg=l2_reg)
    # Add final BN-Relu
    x = BatchNormalization(gamma_regularizer=l2(l2_reg),
                           beta_regularizer=l2(l2_reg),
                           name='trans_final/bn')(x)
    x = Activation('relu', name='trans_final/relu')(x)
    # Global average pooling
    x = GlobalAveragePooling2D(name='fc1')(x)
    x = Dense(nb_classes, kernel_regularizer=l2(l2_reg), activation='softmax', use_bias=False, name='fc2')(x)

    model = Model(inputs=inputs, outputs=x)
    return model

def preprocess_data(data_set):
    mean = np.array([125.3, 123.0, 113.9])
    std = np.array([63.0, 62.1, 66.7])

    data_set -= mean
    data_set /= std
    return data_set


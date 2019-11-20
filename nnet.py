# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 15:23:31 2016

@author: jcsilva
"""

import os

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.layers import TimeDistributed, Bidirectional
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from feats import get_egs

from config import EMBEDDINGS_DIMENSION, MIN_MIX, MAX_MIX
from config import NUM_RLAYERS, SIZE_RLAYERS
from config import BATCH_SIZE, STEPS_PER_EPOCH, NUM_EPOCHS, VALIDATION_STEPS
from config import DEEPC_BASE, DROPOUT, RDROPOUT, L2R, CLIPNORM



def get_dims(generator, embedding_size):
    inp, out = next(generator)
    k = MAX_MIX
    inp_shape = (None, inp['input'].shape[-1])
    out_shape = list(out['kmeans_o'].shape[1:])
    out_shape[-1] *= float(embedding_size)/k
    out_shape[-1] = int(out_shape[-1])
    out_shape = tuple(out_shape)

    return inp_shape, out_shape


def save_model(model, filename):

    # serialize model to JSON
    path_json = os.path.join(DEEPC_BASE, filename + '.json')
    model_json = model.to_json()
    with open(path_json, "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    path_weights = os.path.join(DEEPC_BASE, filename + '.h5')
    model.save_weights(path_weights)
    print("Model saved to disk")


def load_model(filename):

    # load json and create model
    path_json = os.path.join(DEEPC_BASE, filename + '.json')
    json_file = open(path_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    path_weights = os.path.join(DEEPC_BASE, filename + '.h5')
    loaded_model.load_weights(path_weights)
    print("Model loaded from disk")
    return loaded_model


def affinitykmeans(Y, V):
    def norm(tensor):
        square_tensor = K.square(tensor)
        frobenius_norm2 = K.sum(square_tensor, axis=(1, 2))
        return frobenius_norm2

    def dot(x, y):
        return K.batch_dot(x, y, axes=(2, 1))

    def T(x):
        return K.permute_dimensions(x, [0, 2, 1])

    V = K.l2_normalize(K.reshape(V, [BATCH_SIZE, -1,
                                     EMBEDDINGS_DIMENSION]), axis=-1)
    Y = K.reshape(Y, [BATCH_SIZE, -1, MAX_MIX])

    silence_mask = K.sum(Y, axis=2, keepdims=True)
    V = silence_mask * V

    return norm(dot(T(V), V)) - norm(dot(T(V), Y)) * 2 + norm(dot(T(Y), Y))


def train_nnet(train_list, valid_list, weights_path=None):
    train_gen = get_egs(train_list,
                        min_mix=MIN_MIX,
                        max_mix=MAX_MIX,
                        batch_size=BATCH_SIZE)
    valid_gen = get_egs(valid_list,
                        min_mix=MIN_MIX,
                        max_mix=MAX_MIX,
                        batch_size=BATCH_SIZE)
    inp_shape, out_shape = get_dims(train_gen,
                                    EMBEDDINGS_DIMENSION)

    inp = Input(shape=inp_shape, name='input')
    x = inp
    for i in range(NUM_RLAYERS):
        x = Bidirectional(LSTM(SIZE_RLAYERS, return_sequences=True,
                               kernel_regularizer=l2(L2R),
                               recurrent_regularizer=l2(L2R),
                               bias_regularizer=l2(L2R),
                               dropout=DROPOUT,
                               recurrent_dropout=RDROPOUT),
                          input_shape=inp_shape)(x)
    kmeans_o = TimeDistributed(Dense(out_shape[-1],
                                     activation='tanh',
                                     kernel_regularizer=l2(L2R),
                                     bias_regularizer=l2(L2R)),
                               name='kmeans_o')(x)

    model = Model(inputs=[inp], outputs=[kmeans_o])
    if weights_path:
        model.load_weights(weights_path)
    model.compile(loss={'kmeans_o': affinitykmeans},
                  optimizer=Nadam(clipnorm=CLIPNORM))

    # checkpoint
    filepath = os.path.join(DEEPC_BASE, "weights-improvement-{epoch:02d}-{val_loss:.2f}.h5")
    checkpoint = ModelCheckpoint(filepath,
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='min',
                                 save_weights_only=True)
    tb = TensorBoard()
    callbacks_list = [checkpoint, tb]

    model.fit_generator(train_gen,
                        validation_data=valid_gen,
                        validation_steps=VALIDATION_STEPS,
                        steps_per_epoch=STEPS_PER_EPOCH,
                        epochs=NUM_EPOCHS,
                        max_queue_size=512,
                        callbacks=callbacks_list)
    save_model(model, 'model')

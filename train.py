# -*- coding: utf-8 -*-
"""
@author: Kevin Groot Lipman
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"
import tensorflow as tf
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import multi_gpu_model
from keras.optimizers import Adam
from asbestosis.data_generator import DataGenerator
import pickle
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from asbestosis.resnet3d import Resnet3DBuilder
import keras.backend as K


class ModelMGPU(Model):
    # https: // github.com / keras - team / keras / issues / 2436  # issuecomment-354882296
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
           serial-model holds references to the weights in the multi-gpu model.
           '''
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)
        else:
            # return Model.__getattribute__(self, attrname)
            return super(ModelMGPU, self).__getattribute__(attrname)


def auc_roc(y_true, y_pred):
    # from https://github.com/keras-team/keras/issues/6050
    # any tensorflow metric
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value


def train():
    # Initialize, set config
    tf.set_random_seed(42)
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    K.set_session(session)

    # Set training parameters
    # n_classes = 2 if softmax binary classification, 1 if sigmoid binary classification
    # n_channels = 2 if anomaly heatmaps are used, 1 otherwise.
    shape = (96, 192, 192)
    params = {'shape': shape,
              'n_classes': 2,
              'n_channels': 2,
              'label_type': 'soft'}

    filename = '/DATA/kevin/models/resnet_{}_{}'.format(str(shape), params['label_type'])

    # Load soft or hard labels
    with open('/DATA/kevin/partitions/partition_{}.pkl'.format(params['label_type']), 'rb') as file:
        partition, labels = pickle.load(file)
    # Building model, multi_input is used for lung function test (lft) integration
    model = Resnet3DBuilder.build_resnet_18([shape[0], shape[1], shape[2], params['n_channels']],
                                            params['n_classes'],
                                            reg_factor=1e-3,
                                            multi_input=False)

    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=['acc', auc_roc])

    print(model.summary())

    training_generator = DataGenerator(partition['train'], labels, **params, augment=True, batch_size=16, shuffle=True)
    validation_generator = DataGenerator(partition['val'], labels, **params, augment=False, batch_size=16, shuffle=True)

    # Callbacks
    # checkpoint saves best model
    # early stopping stops training if validation loss does not improve for 30 epochs
    checkpoint_loss = ModelCheckpoint(filename + '_loss.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                      mode='auto')
    es = EarlyStopping(monitor='val_loss', mode='auto', verbose=0, patience=30, min_delta=0.001,
                       restore_best_weights=False)
    callbacks_list = [checkpoint_loss, es]

    # Fit model with generators, use_multiprocessing=True lead to crashes
    history = model.fit_generator(generator=training_generator,
                                  validation_data=validation_generator,
                                  validation_steps=len(partition['val']) // validation_generator.batch_size,
                                  epochs=200,
                                  verbose=1,
                                  use_multiprocessing=False,
                                  workers=training_generator.batch_size,
                                  callbacks=callbacks_list)
    model.save(filename + '_final.h5')

    with open('/DATA/kevin/pickle/resnet.pkl', 'wb') as file:
        pickle.dump([history.history], file)


if __name__ == "__main__":
    train()
import os
os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from asbestosis.vae import vae
from asbestosis.data_generator_vae import DataGenerator
import keras.callbacks as clb
import numpy as np
import pickle

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


class VariableScheduler(clb.Callback):
    """Callback used to change loss weights during training"""

    def __init__(self, weight, schedule_fn):
        self.weight = weight
        self.schedule_fn = schedule_fn
        super(VariableScheduler, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        K.set_value(self.weight,
                    self.schedule_fn(K.get_value(self.weight), epoch))
        print('Weight updated to:' + str(K.get_value(self.weight)))


def train():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    K.set_session(session)

    params = {'shape': (256, 256),
              'batch_size': 48,
              'n_channels': 1,
              'shuffle': True}

    with open('/DATA/kevin/partitions/partition_vae.pkl', 'rb') as file:
        train_ID, val_ID = pickle.load(file)

    training_generator = DataGenerator(train_ID, **params)
    validation_generator = DataGenerator(val_ID, **params)

    latent_dim = 4
    latent_fil = 4
    latent = 'dense'
    ann_w = K.variable(0.00)
    sche_fn = lambda weight, epoch: np.min((weight + 0.05, 1))
    sche = VariableScheduler(ann_w, sche_fn)

    filepath = '/DATA/kevin/vae_' + str(latent_dim) + '_' + str(latent_fil) + '_' + str(latent) + '.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

    model = vae(shape=[256, 256, 1],
                annealing_weight=ann_w,
                nr_blocks=6,
                latent=latent,
                latent_dim=latent_dim,
                latent_fil=latent_fil)

    if os.path.isfile('/DATA/kevin/fvae_z' + str(latent_dim) + '_' + str(latent_fil) + '_' + str(latent) + '.h5'):
        model.load_weights('/DATA/kevin/fvae_z' + str(latent_dim) + '_' + str(latent_fil) + '_' + str(latent) + '.h5')
        print('LOADED THE WEIGHTS')

    history = model.fit_generator(generator=training_generator,
                                  validation_data=validation_generator,
                                  epochs=200,
                                  verbose=1,
                                  steps_per_epoch=len(train_ID) / 10 // (params['batch_size']),
                                  callbacks=[checkpoint, sche])

    model.save('/DATA/kevin/vae_' + str(latent_dim) + '_' + str(latent_fil) + '_' + str(latent) + '_final.h5')

    with open('/DATA/kevin/vae_' + str(latent_dim) + '_' + str(latent_fil) + '_' + str(latent) + '.pkl', 'wb') as file:
        pickle.dump(history.history, file)


if __name__ == '__main__':
    train()
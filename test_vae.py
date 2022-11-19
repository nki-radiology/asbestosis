from keras.models import load_model
import numpy as np
import os
from asbestosis.data_generator import DataGenerator
import matplotlib.pyplot as plt
import keras_contrib
from keras.models import Model
import pickle
import glob
from itertools import chain
import nrrd


def plot(data, name, fontsize_ticks=14, fontsize=20):
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_tick_params(width=1)
    ax.yaxis.set_tick_params(width=1)
    plt.hist(data, bins='auto')
    plt.xlabel('{} value'.format(name), fontsize=fontsize)
    plt.ylabel('Frequency', fontsize=fontsize)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.savefig('/DATA/kevin/results/images/{}.png'.format(name))


def predict(plot_distribution=True):
    model = load_model('Z:/active_Kevin/models/vae_1_128_dense.h5', custom_objects={
        'SubPixelUpscaling': keras_contrib.layers.convolutional.subpixelupscaling.SubPixelUpscaling}, compile=False)

    params = {'shape': (96, 256, 256)}

    with open('/DATA/kevin/partitions/partition_vae.pkl', 'rb') as file:
        partition, labels = pickle.load(file)

    list_ID = glob.glob('/DATA/kevin/ct/*')

    test_generator = DataGenerator(list_ID, labels, **params, batch_size=1, shuffle=False, augment=False)

    for item in list_ID:
        x_val, y = test_generator._DataGenerator__data_generation([item])
        x_val = x_val.squeeze(axis=0)
        y_pred = model.predict(x_val)
        y_se = np.abs(np.squeeze(x_val) - np.squeeze(y_pred))**2
        nrrd.write('/DATA/kevin/anomaly_heatmap/' + item + '.nrrd', y_se, index_order='C')

    if plot_distribution:
        ## Explore distribution mean and std tensor
        z_means = []
        z_vars = []
        z_xs = []
        encoder_mean = Model(input=model.layers[0].input, output=model.layers[38].output)
        encoder_var = Model(input=model.layers[0].input, output=model.layers[39].output)
        encoder_z = Model(input=model.layers[0].input, output=model.layers[40].output)

        for i, item in enumerate(list_ID):
            x_val, y = test_generator._DataGenerator__data_generation([item])
            x_val = x_val.squeeze(axis=0)

            z_means.append(encoder_mean.predict(x_val).flatten())
            z_vars.append(encoder_var.predict(x_val).flatten())
            z_xs.append(encoder_z.predict(x_val).flatten())
            print(i)

        z_means = list(chain.from_iterable(z_means))
        z_vars = list(chain.from_iterable(z_vars))
        z_xs = list(chain.from_iterable(z_xs))

        plot(z_means, 'z_means')
        plot(z_vars, 'z_vars')
        plot(z_xs, 'z_xs')

        with open('Z:/active_Kevin/models/vae_1_128_dense.pkl', 'rb') as file:
            history = pickle.load(file)

        plt.figure()
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.ylim([0, 3])
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

if __name__ == '__main__':
    predict(plot_distribution=True)

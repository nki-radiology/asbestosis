"""
Adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
"""

import numpy as np
import keras
import random
from keras.utils import to_categorical
from skimage.transform import resize
from scipy.ndimage import rotate
import nrrd


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, list_IDs, labels, batch_size=2, shape=(96, 128, 128), n_channels=1, lft=None,
                 n_classes=2, shuffle=False, augment=False, label_type='soft'):
        """Initialization"""
        self.shape = shape
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.lft = lft  # lung function test if multiple inputs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment
        self.label_type = label_type
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch

        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)


    def data_augmentation(self, data, rot, ran, alpha=1):
        """ Data augmentation, rotate resize, pad, zoom (alpha) and normalize data randomly"""
        if self.augment:
            if ran[0] < 0.5:
                data = rotate(data, rot, axes=(1, 2))
            if ran[1] < 0.2:
                data = np.flip(data, axis=(0, 2))
            if ran[2] < 0.3:
                alpha = 1 - ran[2]

        # Axial plane is resize with one scale factor, to avoid x-y distortion
        max_dim = np.max((data.shape[1], data.shape[2]))
        zoom_z = [self.shape[0] if data.shape[0] > self.shape[0] else data.shape[0]][0]
        zoom_y = [data.shape[1] * self.shape[1] / max_dim if max_dim > self.shape[1] else data.shape[1]][0]
        zoom_x = [data.shape[2] * self.shape[2] / max_dim if max_dim > self.shape[2] else data.shape[2]][0]
        data = resize(data, (int(np.round(alpha * zoom_z)),
                             int(np.round(alpha * zoom_y)),
                             int(np.round(alpha * zoom_x))), preserve_range=True)

        z_pad = self.shape[0] - data.shape[0]
        y_pad = self.shape[1] - data.shape[1]
        x_pad = self.shape[2] - data.shape[2]

        data = np.pad(data, ((int(np.floor(z_pad / 2)), int(np.ceil(z_pad / 2))),
                             (int(np.floor(y_pad / 2)), int(np.ceil(y_pad / 2))),
                             (int(np.floor(x_pad / 2)), int(np.ceil(x_pad / 2)))),
                      'constant', constant_values=0)
        data_min = 0
        data_max = 1
        data[data > data_max] = data_max
        data[data < data_min] = data_min
        data += -np.min(data)
        data /= np.max(data)
        return data

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.shape, self.n_channels))
        y = np.empty(self.batch_size, dtype=int)
        lft = np.empty(self.batch_size)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            rot = random.choice(range(-10, 10))
            ran = [random.random(), random.random(), random.random()]

            # Load CT image
            ct, _ = nrrd.read('/DATA/kevin/seg_caps/' + str(ID) + '.nrrd', index_order='C')
            ct = self.data_augmentation(data=ct, rot=rot, ran=ran)
            X[i, :, :, :, 0] = ct

            if self.n_channels > 1:
                # Load anomaly heatmap if multiple channels are set
                heatmap, _ = nrrd.read('/DATA/kevin/anomaly_heatmap/' + str(ID) + '.nrrd', index_order='C')
                heatmap = self.data_augmentation(data=heatmap, rot=rot, ran=ran)
                X[i, :, :, :, 1] = heatmap

            if self.lft:
                # Load lung function test if defined as input
                lft[i] = self.lft[ID]
            y[i] = self.labels[ID[0]]

        if self.label_type == 'soft':
            labels = np.array([[1 - s, s] for s in y])
        else:
            labels = to_categorical(np.round(y, 0), 2)
        if self.lft:
            return [X, lft], labels
        else:
            return X, labels

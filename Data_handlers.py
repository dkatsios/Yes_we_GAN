
import matplotlib.pyplot as plt
import os
import random
import numpy as np
np.random.seed(0)
from PIL import Image

import matplotlib.pyplot as plt
from glob import glob

import numpy as np

np.random.seed(0)
dataset_array = None
dataset_list = None
data_shape = None


def make_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path


#mnist dataset
def mnist_data_iterator(batch_size):
    from keras.datasets import mnist
    (X_train, _), (_, _) = mnist.load_data()

    # Rescale -1 to 1
    X_train = X_train / 127.5 - 1.
    X_train = np.expand_dims(X_train, axis=3)

    while True:
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]
        yield imgs


def set_dataset_array(dataset_folder, dataset_array):
    im_paths_list = list(glob(dataset_folder + '*.*'))
    img = np.asarray(Image.open(im_paths_list[0]).convert('RGBA'))
    data_shape = img.shape
    dataset_array = np.zeros((len(im_paths_list), *data_shape))
    for ind, im_path in enumerate(im_paths_list):
        img = Image.open(im_path).convert('RGBA')
        img = np.asarray(img)
        dataset_array[ind] = img
    dataset_array = dataset_array / 127.5 - 1.

    return dataset_array, data_shape


def set_dataset_list(dataset_folder, dataset_list):
    dataset_list = list(glob(dataset_folder + '*.*'))
    img = np.asarray(Image.open(dataset_list[0]).convert('RGBA'))
    data_shape = img.shape

    return dataset_list, data_shape


def folder_data_iterator(batch_size):
    while True:
        idx = np.random.randint(0, dataset_array.shape[0], batch_size)
        imgs = dataset_array[idx]
        yield imgs


def folder_no_load_iterator(batch_size):
    while True:
        imgs = np.zeros((batch_size, *data_shape))

        for i in range(batch_size):
            im_path = random.choice(dataset_list)
            img = Image.open(im_path).convert('RGBA')
            img = np.asarray(img)
            imgs[i] = img
        imgs = imgs / 127.5 - 1.

        yield imgs


def sample_results(epoch, generator, model_params, results_folder, gray=False):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, model_params['latent_dim']))
    gen_imgs = generator.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)

    cmap = None
    if gray:
        cmap = 'gray'
        gen_imgs = np.squeeze(gen_imgs, axis=-1)
    # if gen_imgs.shape[-1] == 4:
    #     gen_imgs = gen_imgs[:, :, :, :-1]

    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt], interpolation='nearest',
                             cmap=cmap)
            axs[i, j].axis('off')
            cnt += 1
    model_labels = '_'.join(model_params['models_labels'])
    fig.savefig("{}/{}_samples_{}.png".format(results_folder, model_labels, epoch))
    plt.close()


def save_models(generator, discriminator, model_params, save_folder, epoch):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    model_labels = '_'.join(model_params['models_labels'])
    generator.save('{}/{}_generator_{}.h5'.format(save_folder, model_labels, epoch))
    discriminator.save('{}/{}_discriminator_{}.h5'.format(save_folder, model_labels, epoch))


def get_data_iterator(dataset, load_full=True):
    global dataset_array, dataset_list, data_shape
    try:
        exec('from keras.datasets import {}'.format(dataset))
        (X_train, _), (_, _) = eval('{}.load_data()'.format(dataset))
        data_shape = X_train[0].shape
        if len(data_shape) == 2:
            data_shape = data_shape[0], data_shape[1], 1
        return mnist_data_iterator, data_shape
    except:
        if not os.path.isdir(dataset):
            raise ValueError('dataset_label must be either a Keras data set or a directory with images')
        if load_full:
            dataset_array, data_shape = set_dataset_array(dataset, dataset_array)
            return folder_data_iterator, data_shape
        else:
            dataset_list, data_shape = set_dataset_list(dataset, dataset_array)
            return folder_no_load_iterator, data_shape



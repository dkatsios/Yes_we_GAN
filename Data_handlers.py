import matplotlib.pyplot as plt
import os
import random
import numpy as np

np.random.seed(0)
from PIL import Image
import cv2

import matplotlib.pyplot as plt
import glob

import numpy as np

np.random.seed(0)
dataset_array = None
dataset_list = None
data_shape = None
keras_dataset_name = None
keras_dataset = None
resize_img = None


def make_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path


# keras dataset
def keras_data_iterator(batch_size):
    global keras_dataset
    # Rescale -1 to 1
    keras_dataset = keras_dataset / 127.5 - 1.
    if len(keras_dataset.shape) == 3:
        keras_dataset = np.expand_dims(keras_dataset, axis=3)

    while True:
        idx = np.random.randint(0, keras_dataset.shape[0], batch_size)
        imgs = keras_dataset[idx]
        yield imgs


def get_im_paths_list(dataset_folder, data_type, search_subfolders=False):
    data_type = '*' if data_type is None else data_type
    rec = '**/' if search_subfolders else ''
    im_paths_list = glob.glob(dataset_folder + '/{}*.{}'.format(rec, data_type), recursive=search_subfolders)
    return im_paths_list


def set_dataset_array(dataset_folder, model_params):
    im_paths_list = get_im_paths_list(dataset_folder, model_params['data_type'], model_params['search_subfolders'])

    try:
        img = cv2.imread(im_paths_list[0], cv2.IMREAD_UNCHANGED)
        if model_params['imgs_resized_size'] is not None:
            img = cv2.resize(img, model_params['imgs_resized_size'])
    except IndexError:
        raise ValueError('There are no files with the specified extension'
                         ' in this directory: {}'.format(dataset_folder))
    data_shape = img.shape
    dataset_array = np.zeros((len(im_paths_list), *data_shape))
    for ind, im_path in enumerate(im_paths_list):
        img = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
        if model_params['imgs_resized_size'] is not None:
            img = cv2.resize(img, model_params['imgs_resized_size'])
        dataset_array[ind] = img / 127.5 - 1.
    dataset_array = dataset_array

    return dataset_array, data_shape


def set_dataset_list(dataset_folder, model_params):
    dataset_list = get_im_paths_list(dataset_folder, model_params['data_type'], model_params['search_subfolders'])
    img = cv2.imread(dataset_list[0], cv2.IMREAD_UNCHANGED)
    if model_params['imgs_resized_size'] is not None:
        img = cv2.resize(img, model_params['imgs_resized_size'])
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
            img = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
            if resize_img is not None:
                img = cv2.resize(img, resize_img)
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

    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt], interpolation='nearest',
                             cmap=cmap)
            axs[i, j].axis('off')
            cnt += 1
    model_labels = '_'.join(model_params['model_labels'])
    fig.savefig("{}/{}_samples_{}.png".format(results_folder, model_labels, epoch))
    plt.close()


def save_models(generator, discriminator, model_params, save_folder, epoch):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    model_labels = '_'.join(model_params['model_labels'])
    generator.save('{}/{}_generator_{}.h5'.format(save_folder, model_labels, epoch))
    discriminator.save('{}/{}_discriminator_{}.h5'.format(save_folder, model_labels, epoch))


def get_data_iterator(dataset, model_params):
    global dataset_array, dataset_list, data_shape, keras_dataset_name, keras_dataset, resize_img
    resize_img = model_params['imgs_resized_size']
    try:
        exec('from keras.datasets import {}'.format(dataset))
        exec('keras_dataset = {}'.format(dataset))
        keras_dataset_name = dataset
        (keras_dataset, _), (_, _) = eval('{}.load_data()'.format(dataset))
        data_shape = keras_dataset[0].shape
        if len(data_shape) == 2:
            data_shape = data_shape[0], data_shape[1], 1
        return keras_data_iterator, data_shape
    except:
        if not os.path.isdir(dataset):
            raise ValueError('dataset_label must be either a Keras data set or a directory with images')

        if model_params['load_full_dataset']:
            dataset_array, data_shape = set_dataset_array(dataset, model_params)
            return folder_data_iterator, data_shape
        else:
            dataset_list, data_shape = set_dataset_list(dataset, model_params)
            return folder_no_load_iterator, data_shape

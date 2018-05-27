
import matplotlib.pyplot as plt
import os
import numpy as np
np.random.seed(0)
from PIL import Image

import matplotlib.pyplot as plt
from glob import glob

import numpy as np

np.random.seed(0)
dataset_array = None


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


def folder_data_iterator(batch_size):
    while True:
        idx = np.random.randint(0, dataset_array.shape[0], batch_size)
        imgs = dataset_array[idx]
        yield imgs


def sample_results(epoch, generator, dimensions, results_folder, gray=False):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, dimensions['latent_dim']))
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
    fig.savefig("{}/{}.png".format(results_folder, epoch))
    plt.close()


def save_models(generator, discriminator, save_folder, epoch):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    generator.save('{}/generator_{}.h5'.format(save_folder, epoch))
    discriminator.save('{}/discriminator_{}.h5'.format(save_folder, epoch))


def get_data_iterator(dataset, load_full=True):
    global dataset_array

    if dataset == 'mnist':
        data_shape = 28, 28, 1
        return mnist_data_iterator, data_shape

    if load_full:
        dataset_array, data_shape = set_dataset_array(dataset, dataset_array)
        return folder_data_iterator, data_shape




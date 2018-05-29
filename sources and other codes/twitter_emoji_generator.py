from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from PIL import Image

import matplotlib.pyplot as plt
from glob import glob

from GAN_template import GAN

import numpy as np
np.random.seed(0)

im_dim = 16
num_channels = 4
dataset_folder = './twitter_dataset/{}x{}/'.format(im_dim, im_dim)


def get_dataset(dataset_folder):
    im_paths_list = list(glob(dataset_folder + '*.png'))
    dataset = np.zeros((len(im_paths_list), im_dim, im_dim, num_channels))
    for ind, im_path in enumerate(im_paths_list):
        img = Image.open(im_path).convert('RGBA')
        dataset[ind] = np.asarray(img)
    return dataset


def data_iterator(batch_size, dimensions):
    dataset = get_dataset(dataset_folder)

    dataset = dataset / 127.5 - 1.

    while True:
        idx = np.random.randint(0, dataset.shape[0], batch_size)
        imgs = dataset[idx]
        yield imgs


def build_generator(dimensions):

    model = Sequential()

    model.add(Dense(256, input_dim=dimensions['latent_dim']))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(dimensions['data_shape']), activation='tanh'))
    model.add(Reshape(dimensions['data_shape']))

    model.summary()

    noise = Input(shape=(dimensions['latent_dim'],))
    img = model(noise)

    return Model(noise, img)


def build_discriminator(dimensions):

    model = Sequential()

    model.add(Flatten(input_shape=dimensions['data_shape']))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    img = Input(shape=dimensions['data_shape'])
    validity = model(img)

    return Model(img, validity)


def sample_results(epoch, generator):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, 100))
    gen_imgs = generator.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig("twitter_results/%d.png" % epoch)
    plt.close()


if __name__ == '__main__':
    data_shape = im_dim, im_dim, num_channels
    latent_dim = 100

    dimensions = {'data_shape': data_shape,
                  'latent_dim': latent_dim}

    models = {'generator': build_generator(dimensions),
              'discriminator': build_discriminator(dimensions)}

    train_params = {'epochs': 100000,
                    'batch_size': 32,
                    'sample_interval': 1000}

    optimizer = Adam(0.0002, 0.5)

    gan = GAN(dimensions, models, optimizer=optimizer)
    gan.train(data_iterator, sample_results, **train_params)

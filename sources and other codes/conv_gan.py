from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from PIL import Image

from keras import initializers
import matplotlib.pyplot as plt
from glob import glob

from GAN_template import GAN

import numpy as np

np.random.seed(0)

im_dim = 32
num_channels = 4
latent_sqrt = 10

dataset_folder = './flags/{}x{}/'.format(im_dim, im_dim)
# results_folder = './flags_results_conv/{}x{}/'.format(im_dim, im_dim)
results_folder = './test_images/'


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
    im_dim = dimensions['data_shape'][0]
    if im_dim == 28:
        init = 7
        blocks = 1
    elif im_dim % 16 == 0:
        init = 4
        blocks = im_dim // 16

    generator = Sequential()
    generator.add(Dense(init * init * 128, input_dim=dimensions['latent_dim'],
                        kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    generator.add(LeakyReLU(0.2))
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(Dropout(.3))
    generator.add(Reshape((init, init, 128)))
    generator.add(UpSampling2D(size=(2, 2)))

    for block in range(blocks):
        generator.add(Conv2D(64, kernel_size=(5, 5), padding='same'))
        generator.add(LeakyReLU(0.2))
        generator.add(BatchNormalization(momentum=0.8))
        generator.add(Dropout(.3))
        generator.add(UpSampling2D(size=(2, 2)))

    generator.add(Conv2D(dimensions['data_shape'][-1], kernel_size=(5, 5),
                         padding='same', activation='tanh'))
    # generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
    # model = Sequential()
    #
    # model.add(Reshape((dimensions['latent_sqrt'], dimensions['latent_sqrt'], 1),
    #                   input_shape=(dimensions['latent_dim'],)))
    # model.add(Conv2D(16, 3, strides=(1, 1), padding='valid'))
    #
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(BatchNormalization(momentum=0.8))
    # model.add(Dropout(.3))
    # # model.add(MaxPooling2D())
    #
    # model.add(Conv2D(32, 3, strides=(2, 2), padding='valid'))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(BatchNormalization(momentum=0.8))
    # model.add(Dropout(.3))
    # # model.add(MaxPooling2D())
    #
    # model.add(Conv2D(64, 5, strides=(2, 2), padding='valid'))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(BatchNormalization(momentum=0.8))
    # model.add(Dropout(.3))
    # # model.add(MaxPooling2D())
    # # model.add(UpSampling2D())
    #
    # model.add(Flatten())
    # model.add(Dense(np.prod(dimensions['data_shape']), activation='tanh'))
    # model.add(Reshape(dimensions['data_shape']))
    # model.summary()
    generator.summary()

    noise = Input(shape=(dimensions['latent_dim'],))
    img = generator(noise)

    return Model(noise, img)


def build_discriminator(dimensions):
    discriminator = Sequential()
    discriminator.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same',
                             input_shape=dimensions['data_shape'],
                             kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same'))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Flatten())
    discriminator.add(Dense(1, activation='sigmoid'))

    # model = Sequential()
    # model.add(Flatten(input_shape=dimensions['data_shape']))
    # model.add(Dense(512))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Dense(256))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Dense(1, activation='sigmoid'))
    # model.summary()

    discriminator.summary()
    img = Input(shape=dimensions['data_shape'])
    validity = discriminator(img)

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
            axs[i, j].imshow(gen_imgs[cnt])
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(results_folder + "%d.png" % epoch)
    plt.close()


if __name__ == '__main__':
    data_shape = im_dim, im_dim, num_channels
    latent_dim = latent_sqrt * latent_sqrt

    dimensions = {'data_shape': data_shape,
                  'latent_dim': latent_dim,
                  'latent_sqrt': latent_sqrt}

    models = {'generator': build_generator(dimensions),
              'discriminator': build_discriminator(dimensions)}

    train_params = {'epochs': 100000,
                    'batch_size': 4,
                    'sample_interval': 200,
                    'gen_to_disc_ratio': 10}  # integer

    optimizer = Adam(0.0002, 0.5)

    gan = GAN(dimensions, models, save_folder=results_folder, dataset=dataset_folder, optimizer=optimizer)
    gan.train(data_iterator, sample_results, **train_params)

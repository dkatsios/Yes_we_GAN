from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

from GAN_template import GAN

import numpy as np
np.random.seed(0)


def data_iterator(batch_size, dimensions):
    (X_train, _), (_, _) = mnist.load_data()

    # Rescale -1 to 1
    X_train = X_train / 127.5 - 1.
    X_train = np.expand_dims(X_train, axis=3)

    while True:
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]
        yield imgs


def build_generator(dimensions):
    model = Sequential()

    model.add(Reshape((dimensions['latent_sqrt'], dimensions['latent_sqrt'], 1),
                      input_shape=(dimensions['latent_dim'],)))
    model.add(Conv2D(16, 3, strides=(1, 1), padding='same'))
    # model.add(Dense(256, input_dim=dimensions['latent_dim']))

    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dropout(.3))
    # model.add(MaxPooling2D())

    model.add(Conv2D(32, 3, strides=(2, 2), padding='same'))
    # model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dropout(.3))
    # model.add(MaxPooling2D())

    model.add(Conv2D(64, 5, strides=(2, 2), padding='same'))
    # model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dropout(.3))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(np.prod(dimensions['data_shape']), activation='tanh'))
    model.add(Reshape(dimensions['data_shape']))

    model.summary()

    noise = Input(shape=(dimensions['latent_dim'],))
    img = model(noise)

    # model = Sequential()
    #
    # model.add(Dense(256, input_dim=dimensions['latent_dim']))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(BatchNormalization(momentum=0.8))
    # model.add(Dense(512))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(BatchNormalization(momentum=0.8))
    # model.add(Dense(1024))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(BatchNormalization(momentum=0.8))
    # model.add(Dense(np.prod(dimensions['data_shape']), activation='tanh'))
    # model.add(Reshape(dimensions['data_shape']))
    #
    # model.summary()
    #
    # noise = Input(shape=(dimensions['latent_dim'],))
    # img = model(noise)

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
            axs[i,j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("mnist_results/%d.png" % epoch)
    plt.close()


if __name__ == '__main__':
    data_shape = 28, 28, 1
    latent_dim = 100

    dimensions = {'data_shape': data_shape,
                  'latent_dim': latent_dim}
    models = {'generator': build_generator(dimensions),
              'discriminator': build_discriminator(dimensions)}
    optimizer = Adam(0.0002, 0.5)

    gan = GAN(dimensions, models, optimizer=optimizer)
    gan.train(data_iterator, sample_results, epochs=30000, batch_size=32, sample_interval=200)

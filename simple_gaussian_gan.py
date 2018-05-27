from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import sys

import numpy as np
np.random.seed(0)


class GAN:
    def __init__(self, dimensions, models, optimizer=Adam(0.0002, 0.5)):
        self.input_shape = dimensions['input_shape']
        self.latent_dim = dimensions['latent_dim']
        self.output_shape = dimensions['input_shape']
        self.inner_batch = dimensions['inner_batch']
        self.generated_shape = dimensions['generated_shape']

        # Build and compile the discriminator
        self.discriminator = models['discriminator']
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = models['generator']

        # The generator takes noise as input and generates imgs
        z = Input(shape=(latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def train(self, data_iterator, sample_results, dimensions, use_uniform=False,
              epochs=10000, batch_size=128, sample_interval=50):

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        data_iter = data_iterator(batch_size, self.input_shape)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            imgs = data_iter.__next__()

            if use_uniform:
                noise = np.random.uniform(0, 1, (batch_size, self.latent_dim))
            else:
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_imgs = get_generated_data(self.generator, noise,
                                          batch_size, self.inner_batch,
                                          self.generated_shape)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0 and sample_results is not None:
                sample_results(epoch, self.generator, batch_size, dimensions)


def data_iterator(batch_size, input_shape):
    while True:
        data = np.random.normal(3, 1, (batch_size, *input_shape))
        print('data shape:', data.shape)
        yield data


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
    model.add(Dense(np.prod(dimensions['generated_shape'])))
    model.add(Reshape(dimensions['generated_shape']))

    model.summary()

    noise = Input(shape=(dimensions['latent_dim'],))
    img = model(noise)

    return Model(noise, img)


def build_discriminator(dimensions):

    model = Sequential()

    model.add(Flatten(input_shape=dimensions['input_shape']))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    img = Input(shape=dimensions['input_shape'])
    validity = model(img)

    return Model(img, validity)


def get_generated_data(generator, noise, batch_size, inner_batch, generated_shape):
    # return generator.predict(noise)
    generated_data = np.zeros((batch_size, inner_batch, *generated_shape))
    print('generated_data shape: ', generated_data.shape)
    return generated_data


def make_histogram(data_real, data_fake, i, n_bins=100):
    fig = plt.figure()
    axes = plt.subplot(111)
    n, bins, patches = plt.hist(data_real, n_bins, normed=1, facecolor='green',
                                alpha=0.75)
    n, bins, patches = plt.hist(data_fake, bins, normed=1, facecolor='blue',
                                alpha=0.75)
    axes.set_xlim(-2, 5)
    axes.set_ylim(0, 1)
    fig.savefig("simple_gaussian/iteration_{}.png".format(i+1))
    plt.close(fig)


def sample_results(epoch, generator, batch_size, dimensions):
    or_gaussian = np.random.normal(3, 1, (dimensions['sample_size'], *dimensions['generated_shape']))
    gen_gaussian = np.zeros(or_gaussian.shape)
    for i in range(dimensions['sample_size'] // dimensions['inner_batch']):
        noise = np.random.uniform(0, 1, (dimensions['inner_batch'], latent_dim))
        batch_gaussian = generator.predict(noise)
        print(batch_gaussian.shape)
        gen_gaussian[i*batch_gaussian.shape[0]:(i+1)*batch_gaussian.shape[0]] = batch_gaussian
    make_histogram(or_gaussian, gen_gaussian, i, n_bins=100)

    # r, c = 5, 5
    # noise = np.random.normal(0, 1, (r * c, 100))
    # gen_imgs = generator.predict(noise)
    #
    # # Rescale images 0 - 1
    # gen_imgs = 0.5 * gen_imgs + 0.5
    #
    # fig, axs = plt.subplots(r, c)
    # cnt = 0
    # for i in range(r):
    #     for j in range(c):
    #         axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
    #         axs[i,j].axis('off')
    #         cnt += 1
    # fig.savefig("images/%d.png" % epoch)
    # plt.close()


if __name__ == '__main__':
    input_shape = 20, 1
    generated_shape = 1,
    latent_dim = 5
    inner_batch = 20
    sample_size = 10000

    dimensions = {'input_shape': input_shape,
                  'latent_dim': latent_dim,
                  'inner_batch': inner_batch,
                  'generated_shape': generated_shape,
                  'sample_size': sample_size}

    models = {'generator': build_generator(dimensions),
              'discriminator': build_discriminator(dimensions)}
    optimizer = Adam(0.0002, 0.5)

    gan = GAN(dimensions, models, optimizer=optimizer)
    gan.train(data_iterator, sample_results, dimensions, use_uniform=True,
              epochs=30000, batch_size=1, sample_interval=200)

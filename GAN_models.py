from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
from keras.models import Sequential, Model
from keras import initializers

import numpy as np
from math import log


# simple gan
def s_gan_generator(dimensions, kernel_initializer):
    generator = Sequential()

    generator.add(Dense(256, input_dim=dimensions['latent_dim'],
                        kernel_initializer=kernel_initializer))
    generator.add(LeakyReLU(alpha=0.2))
    # generator.add(BatchNormalization(momentum=0.8))

    generator.add(Dense(512))
    generator.add(LeakyReLU(alpha=0.2))
    # generator.add(BatchNormalization(momentum=0.8))

    if dimensions['data_shape'][0] <= 64:
        generator.add(Dense(1024))
        generator.add(LeakyReLU(alpha=0.2))
        # generator.add(BatchNormalization(momentum=0.8))

    generator.add(Dense(np.prod(dimensions['data_shape']), activation='tanh'))
    generator.add(Reshape(dimensions['data_shape']))

    generator.summary()

    noise = Input(shape=(dimensions['latent_dim'],))
    img = generator(noise)

    return Model(noise, img)


def s_gan_discriminator(dimensions, kernel_initializer, final_act='sigmoid'):
    discriminator = Sequential()

    discriminator.add(Flatten(input_shape=dimensions['data_shape']))

    # if dimensions['data_shape'][0] <= 64:
    #     discriminator.add(Dense(1024, kernel_initializer=kernel_initializer))
    #     discriminator.add(LeakyReLU(0.2))
    #     discriminator.add(Dropout(.3))

    discriminator.add(Dense(512, kernel_initializer=kernel_initializer))
    discriminator.add(LeakyReLU(0.2))
    # discriminator.add(Dropout(.3))

    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(0.2))
    # discriminator.add(Dropout(.3))

    discriminator.add(Dense(1, activation=final_act))
    discriminator.summary()

    img = Input(shape=dimensions['data_shape'])
    validity = discriminator(img)

    return Model(img, validity)


# convolution gan
def cv_gan_generator(dimensions, kernel_initializer):
    im_dim = dimensions['data_shape'][0]
    if im_dim == 28:
        init = 7
        blocks = 1
    elif im_dim % 16 == 0:
        init = 4
        blocks = int(log(im_dim, 2) - 3)
        print(blocks)
    else:
        print('invalid image size')
        return

    def gen_block(input_shape, block_num):
        """The block is expected to return tensor with x2 spatial dims than the input tensor."""
        block = Sequential(name='gen_block_{}'.format(block_num))

        block.add(Conv2D(64, kernel_size=(5, 5), padding='same', input_shape=input_shape))
        block.add(LeakyReLU(0.2))
        block.add(BatchNormalization(momentum=0.8))
        block.add(Dropout(.3))
        block.add(UpSampling2D(size=(2, 2)))

        return block

    generator = Sequential()
    generator.add(Dense(init * init * 128, input_dim=dimensions['latent_dim'],
                        kernel_initializer=kernel_initializer))
    generator.add(LeakyReLU(0.2))
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(Dropout(.3))
    generator.add(Reshape((init, init, 128)))
    generator.add(UpSampling2D(size=(2, 2)))
    print(generator.output_shape)
    for block_num in range(blocks):
        generator.add(gen_block(generator.output_shape[1:], block_num))
        # generator.add(Conv2D(64, kernel_size=(5, 5), padding='same'))
        # generator.add(LeakyReLU(0.2))
        # generator.add(BatchNormalization(momentum=0.8))
        # generator.add(Dropout(.3))
        # generator.add(UpSampling2D(size=(2, 2)))

    generator.add(Conv2D(dimensions['data_shape'][-1], kernel_size=(5, 5),
                         padding='same', activation='tanh'))
    generator.summary()

    noise = Input(shape=(dimensions['latent_dim'],))
    img = generator(noise)

    return Model(noise, img)


def cv_gan_discriminator(dimensions, kernel_initializer, final_act='sigmoid'):
    discriminator = Sequential()
    discriminator.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same',
                             input_shape=dimensions['data_shape'],
                             kernel_initializer=kernel_initializer))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same'))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Flatten())
    discriminator.add(Dense(1, activation=final_act))

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


def get_models(labels, dimensions):
    final_act = 'linear' if 'wgan' in labels else 'sigmoid'
    kernel_initializer = initializers.random_normal(stddev=0.2) if 'wgan' in labels else 'glorot_uniform'

    if 's' in labels:
        generator = s_gan_generator(dimensions, kernel_initializer=kernel_initializer)
        discriminator = s_gan_discriminator(dimensions, kernel_initializer=kernel_initializer, final_act=final_act)
    elif 'cv' in labels:
        generator = cv_gan_generator(dimensions, kernel_initializer=kernel_initializer)
        discriminator = cv_gan_discriminator(dimensions, kernel_initializer=kernel_initializer, final_act=final_act)
    else:
        print('unknown models label')
        return

    return {'generator': generator,
            'discriminator': discriminator}

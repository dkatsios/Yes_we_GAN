from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model

from math import log


# convolution gan
def cv_gan_generator(model_params, kernel_initializer):
    im_dim = model_params['data_shape'][0]
    if im_dim == 28:
        init = 7
        blocks = 1
    elif im_dim % 16 == 0:
        init = 4
        blocks = int(log(im_dim, 2) - 3)
    else:
        raise ValueError('invalid image size')

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
    generator.add(Dense(init * init * 128, input_dim=model_params['latent_dim'],
                        kernel_initializer=kernel_initializer))
    generator.add(LeakyReLU(0.2))
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(Dropout(.3))
    generator.add(Reshape((init, init, 128)))
    generator.add(UpSampling2D(size=(2, 2)))
    for block_num in range(blocks):
        generator.add(gen_block(generator.output_shape[1:], block_num))

    generator.add(Conv2D(model_params['data_shape'][-1], kernel_size=(5, 5),
                         padding='same', activation='tanh'))
    generator.summary()

    noise = Input(shape=(model_params['latent_dim'],))
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

    discriminator.summary()
    img = Input(shape=dimensions['data_shape'])
    validity = discriminator(img)

    return Model(img, validity)

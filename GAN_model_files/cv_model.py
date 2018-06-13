from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model

from math import log
from .helpers import handle_batch_norm


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
        raise ValueError('invalid image size. It must be either 28 or multiple of 16')

    def gen_block(input_shape, block_num, use_batch_norm):
        """The block is expected to return tensor with 2x spatial dims than the input tensor."""
        block = Sequential(name='gen_block_{}'.format(block_num))

        block.add(Conv2D(64, kernel_size=(5, 5), padding='same', input_shape=input_shape))

        bn = handle_batch_norm(use_batch_norm, is_generator=True)
        if bn is not None:
            block.add(bn)

        block.add(LeakyReLU(0.2))
        block.add(Dropout(.3))
        block.add(UpSampling2D(size=(2, 2)))

        return block

    generator = Sequential()
    generator.add(Dense(init * init * 128, input_dim=model_params['latent_dim'],
                        kernel_initializer=kernel_initializer))

    bn = handle_batch_norm(model_params['use_batch_norm'], is_generator=True)
    if bn is not None:
        generator.add(bn)

    generator.add(LeakyReLU(0.2))
    generator.add(Dropout(.3))
    generator.add(Reshape((init, init, 128)))
    generator.add(UpSampling2D(size=(2, 2)))
    for block_num in range(blocks):
        generator.add(gen_block(generator.output_shape[1:], block_num, model_params['use_batch_norm']))

    generator.add(Conv2D(model_params['data_shape'][-1], kernel_size=(5, 5),
                         padding='same', activation='tanh'))
    generator.summary()

    noise = Input(shape=(model_params['latent_dim'],))
    img = generator(noise)

    return Model(noise, img)


def cv_gan_discriminator(model_params, kernel_initializer, final_act='sigmoid'):
    discriminator = Sequential()
    discriminator.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same',
                             input_shape=model_params['data_shape'],
                             kernel_initializer=kernel_initializer))

    bn = handle_batch_norm(model_params['use_batch_norm'], is_generator=False)
    if bn is not None:
        discriminator.add(bn)

    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same'))

    bn = handle_batch_norm(model_params['use_batch_norm'], is_generator=False)
    if bn is not None:
        discriminator.add(bn)

    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Flatten())
    discriminator.add(Dense(1, activation=final_act))

    discriminator.summary()
    img = Input(shape=model_params['data_shape'])
    validity = discriminator(img)

    return Model(img, validity)

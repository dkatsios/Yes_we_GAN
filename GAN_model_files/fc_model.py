from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model

import numpy as np
from .helpers import handle_batch_norm


# fully connected gan
def fc_gan_generator(model_params, kernel_initializer):
    layers_sizes = [256, 512, 1024]

    if model_params['layers_sizes'] is not None:
        if isinstance(model_params['layers_sizes'], (list, tuple)):
            layers_sizes = model_params['layers_sizes']
        elif isinstance(model_params['layers_sizes'], dict):
            layers_sizes = model_params['layers_sizes']['generator']

    def gen_block(layer_size, input_shape, block_num, use_batch_norm, kernel_initializer):
        block = Sequential(name='gen_block_{}'.format(block_num))

        block.add(Dense(layer_size, input_dim=input_shape, kernel_initializer=kernel_initializer))

        bn = handle_batch_norm(use_batch_norm, is_generator=True)
        if bn is not None:
            block.add(bn)

        block.add(LeakyReLU(alpha=0.2))

        return block

    generator = Sequential()
    for block_num, layer_size in enumerate(layers_sizes):
        if block_num == 0:
            input_shape = model_params['latent_dim']
        else:
            input_shape = generator.output_shape[1]

        layer = gen_block(layer_size, input_shape, block_num, model_params['use_batch_norm'], kernel_initializer)
        generator.add(layer)

    generator.add(Dense(np.prod(model_params['data_shape']), activation='tanh'))
    generator.add(Reshape(model_params['data_shape']))

    generator.summary()

    noise = Input(shape=(model_params['latent_dim'],))
    img = generator(noise)

    return Model(noise, img)


def fc_gan_discriminator(model_params, kernel_initializer, final_act='sigmoid'):
    layers_sizes = [512, 256]

    if model_params['layers_sizes'] is not None:
        if isinstance(model_params['layers_sizes'], (list, tuple)):
            layers_sizes = model_params['layers_sizes']
        elif isinstance(model_params['layers_sizes'], dict):
            layers_sizes = model_params['layers_sizes']['discriminator']

    def disc_block(layer_size, input_shape, block_num, use_batch_norm, kernel_initializer):
        block = Sequential(name='gen_block_{}'.format(block_num))

        block.add(Dense(layer_size, input_dim=input_shape, kernel_initializer=kernel_initializer))

        bn = handle_batch_norm(use_batch_norm, is_generator=False)
        if bn is not None:
            block.add(bn)

        block.add(LeakyReLU(alpha=0.2))

        return block

    discriminator = Sequential()
    discriminator.add(Flatten(input_shape=model_params['data_shape']))

    for block_num, layer_size in enumerate(layers_sizes):
        input_shape = discriminator.output_shape[1]
        layer = disc_block(layer_size, input_shape, block_num, model_params['use_batch_norm'], kernel_initializer)
        discriminator.add(layer)

    discriminator.add(Dense(1, activation=final_act))

    discriminator.summary()

    img = Input(shape=model_params['data_shape'])
    validity = discriminator(img)

    return Model(img, validity)

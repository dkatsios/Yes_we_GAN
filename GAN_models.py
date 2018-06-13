from keras import initializers


def get_kernel_initializer(model_params):
    kernel_initializer = 'glorot_uniform'
    if model_params['wgan']:
        if model_params['wgan_gradient_penalty']:
            kernel_initializer = 'he_normal'
        else:
            kernel_initializer = initializers.random_normal(stddev=0.2)
    if model_params['kernel_initializer'] is not None:
        kernel_initializer = model_params['kernel_initializer']
    return kernel_initializer


def get_models(model_params):
    final_act = 'linear' if model_params['wgan'] else 'sigmoid'
    kernel_initializer = get_kernel_initializer(model_params)
    mt = model_params['model_type']
    exec('from GAN_model_files.{}_model import {}_gan_generator, {}_gan_discriminator'.format(mt, mt, mt))

    generator = eval('{}_gan_generator(model_params, kernel_initializer=kernel_initializer)'.format(mt))
    discriminator = eval('{}_gan_discriminator(model_params, '
                         'kernel_initializer=kernel_initializer, final_act=final_act)'.format(mt))

    return {'generator': generator,
            'discriminator': discriminator}

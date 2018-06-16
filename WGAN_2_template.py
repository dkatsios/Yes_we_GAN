from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
from keras.layers.merge import _Merge

from functools import partial
import numpy as np
import os

np.random.seed(1000)


def wasserstein_loss(y_true, y_pred):
    """Calculates the Wasserstein loss for a sample batch.

    The Wasserstein loss function is very simple to calculate. In a standard GAN, the discriminator
    has a sigmoid output, representing the probability that samples are real or generated. In Wasserstein
    GANs, however, the output is linear with no activation function! Instead of being constrained to [0, 1],
    the discriminator wants to make the distance between its output for real and generated samples as large as possible.

    The most natural way to achieve this is to label generated samples -1 and real samples 1, instead of the
    0 and 1 used in normal GANs, so that multiplying the outputs by the labels will give you the loss immediately.

    Note that the nature of this loss means that it can be (and frequently will be) less than 0."""
    return K.mean(y_true * y_pred)


def gradient_penalty_loss(y_true, y_pred, averaged_samples, wgan_gp_weight):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.

    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the loss function
    that penalizes the network if the gradient norm moves away from 1. However, it is impossible to evaluate
    this function at all points in the input space. The compromise used in the paper is to choose random points
    on the lines between real and generated samples, and check the gradients at these points. Note that it is the
    gradient w.r.t. the input averaged samples, not the weights of the discriminator, that we're penalizing!

    In order to evaluate the gradients, we must first run samples through the generator and evaluate the loss.
    Then we get the gradients of the discriminator w.r.t. the input averaged samples.
    The l2 norm and penalty can then be calculated for this gradient.

    Note that this loss function requires the original averaged samples as input, but Keras only supports passing
    y_true and y_pred to loss functions. To get around this, we make a partial() of the function with the
    averaged_samples argument, and use that for model training."""
    # first get the gradients:
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = wgan_gp_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)


class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this outputs a random point on the line
    between each pair of input points.

    Inheriting from _Merge is a little messy but it was the quickest solution I could think of.
    Improvements appreciated."""

    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.name = 'random_weighted_average'

    def _merge_function(self, inputs):
        weights = K.random_uniform((self.batch_size, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


class WGAN_2:
    def __init__(self, model_params, train_params, models, save_folders,
                 dataset, optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9)):

        self.model_params = model_params
        self.data_shape = model_params['data_shape']
        self.latent_dim = model_params['latent_dim']
        self.output_shape = model_params['data_shape']
        self.save_folders = save_folders
        self.gray = self.data_shape[-1] == 1
        self.epoch = 0

        # Build and compile the discriminator
        self.discriminator = models['discriminator']
        # Build the generator
        self.generator = models['generator']

        self.discriminator.trainable = False
        noise = Input(shape=(self.latent_dim,))
        gen_imgs = self.generator(noise)
        validity_gen = self.discriminator(gen_imgs)
        self.combined = Model(inputs=noise, outputs=validity_gen)
        # We use the Adam paramaters from Gulrajani et al.
        self.combined.compile(optimizer=optimizer, loss=wasserstein_loss)

        # Now that the generator_model is compiled, we can make the discriminator layers trainable.
        self.discriminator.trainable = True
        self.generator.trainable = False

        wgan_gp_weight = self.model_params['wgan_gradient_penalty']

        # The discriminator_model is more complex. It takes both real image samples and random noise seeds as input.
        # The noise seed is run through the generator model to get generated images. Both real and generated images
        # are then run through the discriminator. Although we could concatenate the real and generated images into a
        # single tensor, we don't (see model compilation for why).
        real_imgs = Input(shape=self.data_shape)
        noise = Input(shape=(self.latent_dim,))
        gen_imgs = self.generator(noise)
        gen_validity = self.discriminator(gen_imgs)
        real_validity = self.discriminator(real_imgs)

        # We also need to generate weighted-averages of real and generated samples,
        # to use for the gradient norm penalty.
        avg_imgs = RandomWeightedAverage(train_params['batch_size'])([real_imgs, gen_imgs])
        # We then run these samples through the discriminator as well. Note that we never really use the discriminator
        # output for these samples - we're only running them to get the gradient norm for the gradient penalty loss.
        avg_validity = self.discriminator(avg_imgs)

        # The gradient penalty loss function requires the input averaged samples to get gradients. However,
        # Keras loss functions can only have two arguments, y_true and y_pred. We get around this by making a partial()
        # of the function with the averaged samples here.
        partial_gp_loss = partial(gradient_penalty_loss,
                                  averaged_samples=avg_imgs,
                                  wgan_gp_weight=wgan_gp_weight)
        partial_gp_loss.__name__ = 'gradient_penalty'  # Functions need names or Keras will throw an error

        # Keras requires that inputs and outputs have the same number of samples. This is why we didn't concatenate the
        # real samples and generated samples before passing them to the discriminator: If we had, it would create an
        # output with 2 * batch_size samples, while the output of the "averaged" samples for gradient penalty
        # would have only batch_size samples.

        # If we don't concatenate the real and generated samples, however, we get three outputs: One of the generated
        # samples, one of the real samples, and one of the averaged samples, all of size batch_size. This works neatly!
        self.discriminator_model = Model(inputs=[real_imgs, noise],
                                         outputs=[real_validity,
                                                  gen_validity,
                                                  avg_validity])
        # We use the Adam paramaters from Gulrajani et al. We use the Wasserstein loss for both the real and generated
        # samples, and the gradient penalty loss for the averaged samples.
        self.discriminator_model.compile(optimizer=optimizer,
                                         loss=[wasserstein_loss,
                                               wasserstein_loss,
                                               partial_gp_loss])

    def get_repetitions(self, index):
        reps = 1
        if isinstance(self.gen_to_disc_ratio, (tuple, list)):
            reps = max(1, int(self.gen_to_disc_ratio[index]))
        elif callable(self.gen_to_disc_ratio):
            reps = self.gen_to_disc_ratio(self.epoch)[index]
        return reps

    def train(self, data_iterator, sample_results, save_models, train_params):

        epochs = train_params['epochs']
        batch_size = train_params['batch_size']
        sample_interval = train_params['sample_interval']
        model_interval = train_params['model_interval']
        self.gen_to_disc_ratio = train_params['gen_to_disc_ratio']
        # We make three label vectors for training. positive_y is the label vector for real samples, with value 1.
        # negative_y is the label vector for generated samples, with value -1. The dummy_y vector is passed to the
        # gradient_penalty loss function and is not used.
        positive_y = np.ones((batch_size, 1), dtype=np.float32)
        negative_y = -positive_y
        dummy_y = np.zeros((batch_size, 1), dtype=np.float32)

        data_iter = data_iterator(batch_size)

        for epoch in range(1, epochs + 1):
            self.epoch = epoch

            # ---------------------
            #  Train Discriminator
            # ---------------------
            reps = self.get_repetitions(1)
            for _ in range(reps):
                # Select a random batch of images
                imgs = data_iter.__next__()

                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                d_loss = self.discriminator_model.train_on_batch([imgs, noise],
                                                                 [positive_y, negative_y, dummy_y])

            # ---------------------
            #  Train Generator
            # ---------------------

            reps = self.get_repetitions(0)
            for _ in range(reps):
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                g_loss = self.combined.train_on_batch(noise, positive_y)

                # Plot the progress
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if sample_interval is not None and epoch % sample_interval == 0 and sample_results is not None:
                sample_results(epoch, self.generator, self.model_params,
                               self.save_folders['images_folder'], self.gray)

            if model_interval is not None and epoch % model_interval == 0 and save_models is not None:
                save_models(self.generator, self.discriminator,
                            self.model_params, self.save_folders['models_folder'], epoch)

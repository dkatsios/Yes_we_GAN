from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K

import numpy as np
import os

np.random.seed(1000)


def wasserstein_loss(y_true, y_pred):
    """
    Wasserstein distance for GAN
    author use:
    g_loss = mean(-fake_logit)
    c_loss = mean(fake_logit - true_logit)
    logit just denote result of discrimiantor without activated
    """
    return K.mean(y_true * y_pred)


def clip_weight(weight, lower, upper):
    weight_clip = []
    for w in weight:
        w = np.clip(w, lower, upper)
        weight_clip.append(w)
    return weight_clip


class WGAN:
    def __init__(self, model_params, models, save_folders, dataset, optimizer='RMSprop'):

        self.model_params = model_params
        self.data_shape = model_params['data_shape']
        self.latent_dim = model_params['latent_dim']
        self.output_shape = model_params['data_shape']
        self.save_folders = save_folders
        self.gray = self.data_shape[-1] == 1
        self.epoch = 0

        # Build and compile the discriminator
        self.discriminator = models['discriminator']
        self.discriminator.compile(loss=wasserstein_loss,
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = models['generator']

        # The generator takes noise as input and generates imgs
        z = Input(shape=(model_params['latent_dim'],))
        gen_imgs = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity_gen = self.discriminator(gen_imgs)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity_gen)
        self.combined.compile(loss=wasserstein_loss,
                              optimizer=optimizer)

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
        wgan_clip = train_params['wgan_clip']

        # # Adversarial ground truths
        # valid = np.ones((batch_size, 1)) * -1
        # fake = np.ones((batch_size, 1))

        # Labels for generated and real data
        yDis = np.ones(2 * batch_size)

        # one-sided label smoothing
        yDis[:batch_size] = -1

        yGen = np.ones(batch_size) * -1

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

                # Generate a batch of new images
                gen_imgs = self.generator.predict(noise)

                X = np.concatenate([imgs, gen_imgs])

                # Train the discriminator
                # d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                # d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                # d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                d_loss = self.discriminator.train_on_batch(X, yDis)

                if wgan_clip != -1:
                    # clip weights of discriminator
                    d_weight = self.discriminator.get_weights()
                    d_weight = clip_weight(d_weight, -wgan_clip, wgan_clip)
                    self.discriminator.set_weights(d_weight)

            # ---------------------
            #  Train Generator
            # ---------------------

            reps = self.get_repetitions(0)
            for _ in range(reps):
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Train the generator (to have the discriminator label samples as valid)
                g_loss = self.combined.train_on_batch(noise, yGen)

                # Plot the progress
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if sample_interval is not None and epoch % sample_interval == 0 and sample_results is not None:
                sample_results(epoch, self.generator, self.model_params,
                               self.save_folders['images_folder'], self.gray)

            if model_interval is not None and epoch % model_interval == 0 and save_models is not None:
                save_models(self.generator, self.discriminator,
                            self.model_params, self.save_folders['models_folder'], epoch)


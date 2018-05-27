from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam

import numpy as np
import os
np.random.seed(0)


class GAN:
    def __init__(self, dimensions, models, save_folders, dataset, optimizer=Adam(0.0002, 0.5)):
        self.dimensions = dimensions
        self.data_shape = dimensions['data_shape']
        self.latent_dim = dimensions['latent_dim']
        self.output_shape = dimensions['data_shape']
        self.save_folders = save_folders
        self.gray = dataset == 'mnist'

        # Build and compile the discriminator
        self.discriminator = models['discriminator']
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = models['generator']

        # The generator takes noise as input and generates imgs
        z = Input(shape=(dimensions['latent_dim'],))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def train(self, data_iterator, sample_results, save_models,
              epochs, batch_size=128,
              sample_interval=50, model_interval=200,
              gen_to_disc_ratio=1):

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        data_iter = data_iterator(batch_size)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            reps = max(1, int(1 / gen_to_disc_ratio))
            for _ in range(reps):
                # Select a random batch of images
                imgs = data_iter.__next__()

                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Generate a batch of new images
                gen_imgs = self.generator.predict(noise)

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            reps = max(1, int(gen_to_disc_ratio))
            for _ in range(reps):
                noise = np.random.normal(0, 1, (batch_size, 100))

                # Train the generator (to have the discriminator label samples as valid)
                g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0 and sample_results is not None:
                sample_results(epoch, self.generator, self.dimensions,
                               self.save_folders['results_folder'], self.gray)

            if epoch % model_interval == 0 and save_models is not None:
                save_models(self.generator, self.discriminator, self.save_folders['models_folder'], epoch)

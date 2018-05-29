from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam

import tensorflow as tf
import numpy as np
import numbers
np.random.seed(0)
tf.set_random_seed(0)


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

    def train(self, data_iterator, sample_results, save_models, train_params):

        epochs = train_params['epochs']
        batch_size = train_params['batch_size']
        sample_interval = train_params['sample_interval']
        model_interval = train_params['model_interval']
        gen_to_disc_ratio = train_params['gen_to_disc_ratio']
        label_smoothing = train_params['label_smoothing']

        # # Adversarial ground truths
        # valid = np.ones((batch_size, 1))
        # fake = np.zeros((batch_size, 1))

        data_iter = data_iterator(batch_size)

        for epoch in range(1, epochs + 1):
            # Adversarial ground truths
            real, gen = self.get_valid_vals(label_smoothing, batch_size)
            # real = np.ones((batch_size, 1))
            # gen = np.zeros((batch_size, 1))

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
                d_loss_real = self.discriminator.train_on_batch(imgs, real)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, gen)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            reps = max(1, int(gen_to_disc_ratio))
            for _ in range(reps):
                noise = np.random.normal(0, 1, (batch_size, 100))

                # Train the generator (to have the discriminator label samples as valid)
                g_loss = self.combined.train_on_batch(noise, real)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0 and sample_results is not None:
                sample_results(epoch, self.generator, self.dimensions,
                               self.save_folders['results_folder'], self.gray)

            if epoch % model_interval == 0 and save_models is not None:
                save_models(self.generator, self.discriminator, self.save_folders['models_folder'], epoch)

    @classmethod
    def get_valid_vals(cls, label_smoothing, batch_size):
        real = np.ones((batch_size, 1))
        gen = np.zeros((batch_size, 1))

        if label_smoothing is None:
            return real, gen

        if label_smoothing['real'] is not None:
            real = cls.use_label_smoothing(label_smoothing['real'], batch_size)

        if label_smoothing['gen'] is not None:
            gen = cls.use_label_smoothing(label_smoothing['gen'], batch_size)

        return real, gen

    @staticmethod
    def use_label_smoothing(value, batch_size):
        if isinstance(value, numbers.Number):
            labels = np.ones((batch_size, 1)) * value
        else:
            labels = np.random.uniform(*value, (batch_size, 1))

        return labels


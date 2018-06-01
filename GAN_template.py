from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam

import tensorflow as tf
import numpy as np
import numbers
np.random.seed(0)
tf.set_random_seed(0)


class GAN:
    def __init__(self, model_params, models, save_folders, dataset, optimizer=Adam(0.0002, 0.5)):
        self.model_params = model_params
        self.data_shape = model_params['data_shape']
        self.latent_dim = model_params['latent_dim']
        self.output_shape = model_params['data_shape']
        self.save_folders = save_folders
        self.gray = self.data_shape[-1] == 1
        self.epoch = 0

        # Build and compile the discriminator
        self.discriminator = models['discriminator']
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = models['generator']

        # The generator takes noise as input and generates imgs
        z = Input(shape=(model_params['latent_dim'],))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

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
        label_smoothing = train_params['label_smoothing']

        data_iter = data_iterator(batch_size)

        for epoch in range(1, epochs + 1):
            self.epoch = epoch
            # Adversarial ground truths
            real, gen = self.get_valid_vals(label_smoothing, batch_size)

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

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(imgs, real)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, gen)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

                reps = self.get_repetitions(0)
            g_loss = None
            for _ in range(reps):
                noise = np.random.normal(0, 1, (batch_size, 100))

                # Train the generator (to have the discriminator label samples as valid)
                g_loss = self.combined.train_on_batch(noise, real)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0 and sample_results is not None:
                sample_results(epoch, self.generator, self.model_params,
                               self.save_folders['images_folder'], self.gray)

            if epoch % model_interval == 0 and save_models is not None:
                save_models(self.generator, self.discriminator,
                            self.model_params, self.save_folders['models_folder'], epoch)

    @classmethod
    def get_valid_vals(cls, label_smoothing, batch_size):
        real = np.ones((batch_size, 1))
        gen = np.zeros((batch_size, 1))

        if label_smoothing is None:
            return real, gen

        if 'real' in label_smoothing.keys() and label_smoothing['real'] is not None:
            real = cls.use_label_smoothing(label_smoothing['real'], batch_size)

        if 'gen' in label_smoothing.keys() and label_smoothing['gen'] is not None:
            gen = cls.use_label_smoothing(label_smoothing['gen'], batch_size)

        return real, gen

    @staticmethod
    def use_label_smoothing(value, batch_size):
        if isinstance(value, numbers.Number):
            labels = np.ones((batch_size, 1)) * value
        else:
            labels = np.random.uniform(*value, (batch_size, 1))

        return labels


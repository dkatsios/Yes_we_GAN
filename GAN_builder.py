from keras.optimizers import Adam

from GAN_models import get_models
from Data_handlers import get_data_iterator, sample_results, save_models, make_folder
from GAN_template import GAN
from WGAN_template import WGAN


def build_and_train(models_labels, dataset_label, train_params, save_folders):

    load_full = True
    if 'dont_load_full' in models_labels:
        load_full = False
        models_labels.remove('dont_load_full')
    data_iterator, data_shape = get_data_iterator(dataset_label, load_full=load_full)

    latent_dim = 100

    model_params = {'data_shape': data_shape,
                    'latent_dim': latent_dim,
                    'models_labels': models_labels}

    for folder in save_folders.values():
        make_folder(folder)

    models = get_models(models_labels, model_params)

    if 'wgan' in models_labels:
        gan = WGAN(model_params, models, save_folders, dataset_label, optimizer='RMSprop')
    else:
        gan = GAN(model_params, models, save_folders, dataset_label, optimizer=Adam(0.0002, 0.5))

    gan.train(data_iterator, sample_results, save_models, train_params)


def run():
    """high level function for applying GAN on specific data sets.
    The dataset is defined as an element at the models_labels list
    It is a string that can be either the name of an existing Keras dataset (e.g. mnist)
    or the (relative) path of a folder that contains the (equal sized) train images.
    The other elements of the list can be:

    - one of 'fc' or 'cv' for denoting the type of layers of the generator and the discriminator models.
      fc stands for Fully connected (Dense) while cv for convolutional.

    - 'wgan' for the training method to be Wasserstein
      in this case the train_params['wgan_clip'] value denotes the discriminator weights clipping
      select -1 for no clipping

    - 'ls' for smoothing the labels of the real and the generated data during training
      in this case the train_params['label_smoothing'] value denotes the smoothing approach.
      It is a dict with keys 'real' and 'gen', and values either a number or a tuple of two numbers or None.
       - If None, no smoothing will take place for the specific category
       - If a single number, the labels of this category will be equal to this number.
       - If a tuple of two numbers, the labels will be uniformly sampled from this interval.

    The train_params['gen_to_disc_ratio'] value is used to define the ratio of training circles between
    the generator and the discriminator for each repetition.

    The train_params['gen_to_disc_ratio'] value denotes every how many epochs a number of generated images
    will be sampled and saved.

    The train_params['model_interval'] value denotes every how many epochs the generator and discriminator models
    will be saved.
    """

    models_labels = list()
    train_params = {'epochs': 30000,
                    'batch_size': 64,
                    'sample_interval': 1000,
                    'model_interval': 5000,
                    'wgan_clip': -1,
                    'label_smoothing': None,  # {'real': 1, 'gen': 0},
                    'gen_to_disc_ratio': 1 / 1}

    models_labels.append('fc')
    dataset_label = 'mnist'

    save_folders = {'images_folder': './result_images/{}/{}/'.format(dataset_label, '_'.join(models_labels)),
                    'models_folder': './result_models/{}/{}/'.format(dataset_label, '_'.join(models_labels))}

    build_and_train(models_labels, dataset_label, train_params=train_params, save_folders=save_folders)


if __name__ == '__main__':
    run()

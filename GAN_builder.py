from keras.optimizers import Adam

from GAN_models import get_models
from Data_handlers import get_data_iterator, sample_results, save_models, make_folder
from GAN_template import GAN
from WGAN_template import WGAN

latent_sqrt = 10


def build_and_train(models_label, dataset_label, im_dim, train_params):
    if dataset_label == 'mnist':
        dataset = dataset_label
    else:
        dataset = './{}/{}x{}/'.format(dataset_label, im_dim, im_dim)

    data_iterator, data_shape = get_data_iterator(dataset)
    im_dim = data_shape[0]

    save_results_folder = make_folder(
        './result_images/{}/{}/{}x{}/'.format(dataset_label, models_label, im_dim, im_dim))
    save_models_folder = make_folder(
        './result_models/{}/{}/{}x{}/'.format(dataset_label, models_label, im_dim, im_dim))

    save_folders = {'results_folder': save_results_folder,
                    'models_folder': save_models_folder}
    latent_dim = latent_sqrt ** 2

    dimensions = {'data_shape': data_shape,
                  'latent_dim': latent_dim,
                  'latent_sqrt': latent_sqrt}

    models = get_models(models_label, dimensions)

    if 'w' in models_label:
        gan = WGAN(dimensions, models, save_folders, dataset, clip=-1, optimizer='RMSprop')
    else:
        gan = GAN(dimensions, models, save_folders, dataset, optimizer=Adam(0.0002, 0.5))

    gan.train(data_iterator, sample_results, save_models, **train_params)


if __name__ == '__main__':

    train_params = {'epochs':  30001,
                    'batch_size': 64,
                    'sample_interval':  200,
                    'model_interval':  500,
                    'gen_to_disc_ratio': 1 / 1}

    models_labels = ['s', 'cv', 's_w', 'cv_w']
    im_dims = [32, 64]

    dataset_label = 'mnist'
    im_dim = 28
    for models_label in models_labels:
        build_and_train(models_label, dataset_label, im_dim=im_dim, train_params=train_params)

    dataset_label = 'flags'
    for im_dim in im_dims:
        for models_label in models_labels:
            build_and_train(models_label, dataset_label, im_dim=im_dim, train_params=train_params)

    # models_label = 'cv'
    # dataset_label = 'flags'
    # im_dim = 128
    #
    # build_and_train(models_label, dataset_label, im_dim=im_dim, train_params=train_params)



from keras.optimizers import Adam

from GAN_models import get_models
from Data_handlers import get_data_iterator, sample_results, save_models, make_folder
from GAN_template import GAN
from WGAN_template import WGAN


def build_and_train(models_labels, dataset_label, im_dim, train_params):
    if dataset_label == 'mnist':
        dataset = dataset_label
    else:
        dataset = './{}/{}x{}/'.format(dataset_label, im_dim, im_dim)

    data_iterator, data_shape = get_data_iterator(dataset)
    im_dim = data_shape[0]

    save_results_folder = make_folder(
        './result_images/{}/{}/{}x{}/'.format(dataset_label, '_'.join(models_labels), im_dim, im_dim))
    save_models_folder = make_folder(
        './result_models/{}/{}/{}x{}/'.format(dataset_label, '_'.join(models_labels), im_dim, im_dim))

    save_folders = {'results_folder': save_results_folder,
                    'models_folder': save_models_folder}
    latent_dim = 100

    model_params = {'data_shape': data_shape,
                    'latent_dim': latent_dim,
                    'models_labels': models_labels}

    models = get_models(models_labels, model_params)

    if 'wgan' in models_labels:
        gan = WGAN(model_params, models, save_folders, dataset, optimizer='RMSprop')
    else:
        gan = GAN(model_params, models, save_folders, dataset, optimizer=Adam(0.0002, 0.5))

    gan.train(data_iterator, sample_results, save_models, train_params)


if __name__ == '__main__':
    models_labels = list()
    train_params = {'epochs': 30000,
                    'batch_size': 64,
                    'sample_interval': 200,
                    'model_interval': 500,
                    'wgan_clip': -1,
                    'label_smoothing': {'real': 1, 'gen': None},
                    'gen_to_disc_ratio': 1 / 1}

    model_categories = ['s', 'cv']
    im_dims = [32, 64]

    dataset_label = 'mnist'
    im_dim = 28
    for model_category in model_categories:
        models_labels = list()
        models_labels.append(model_category)
        build_and_train(models_labels, dataset_label, im_dim=im_dim, train_params=train_params)

        models_labels.append('wgan')
        build_and_train(models_labels, dataset_label, im_dim=im_dim, train_params=train_params)

    dataset_label = 'flags'
    for im_dim in im_dims:
        for model_category in model_categories:
            models_labels = list()
            models_labels.append(model_category)
            build_and_train(models_labels, dataset_label, im_dim=im_dim, train_params=train_params)

            models_labels.append('wgan')
            build_and_train(models_labels, dataset_label, im_dim=im_dim, train_params=train_params)

    train_params['label_smoothing']['real'] = .9
    dataset_label = 'mnist'
    im_dim = 28
    for model_category in model_categories:
        models_labels = list()
        models_labels.append(model_category)
        models_labels.append('ls')
        build_and_train(models_labels, dataset_label, im_dim=im_dim, train_params=train_params)

        models_labels.append('wgan')
        build_and_train(models_labels, dataset_label, im_dim=im_dim, train_params=train_params)

    dataset_label = 'flags'
    for im_dim in im_dims:
        for model_category in model_categories:
            models_labels = list()
            models_labels.append(model_category)
            models_labels.append('ls')
            build_and_train(models_labels, dataset_label, im_dim=im_dim, train_params=train_params)

            models_labels.append('wgan')
            build_and_train(models_labels, dataset_label, im_dim=im_dim, train_params=train_params)

    # models_labels.append('cv')
    # models_labels.append('wgan')
    #
    # dataset_label = 'mnist'
    # im_dim = 28
    #
    # build_and_train(models_labels, dataset_label, im_dim=im_dim, train_params=train_params)

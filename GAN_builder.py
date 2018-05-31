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


if __name__ == '__main__':
    models_labels = list()
    train_params = {'epochs': 30000,
                    'batch_size': 64,
                    'sample_interval': 1000,
                    'model_interval': 5000,
                    'wgan_clip': -1,  # -1 for no clipping
                    'label_smoothing': None,  # {'real': 1, 'gen': 0},
                    'gen_to_disc_ratio': 1 / 1}

    models_labels.append('fc')  # append 'fc' for fully connected or 'cv' for convolutional model
    # models_labels.append('wgan')  # uncomment for using wgan as training method
    # models_labels.append('ls')  # uncomment for denoting label smoothing at the output files' names

    dataset_label = './flags/32x32/'  # if data set is a folder with images, write the path of the folder
    # dataset_label = 'fashion_mnist'  # if data set is a folder with images, write the path of the folder

    save_folders = {'images_folder': './result_images/{}/{}/'.format(dataset_label, '_'.join(models_labels)),
                    'models_folder': './result_models/{}/{}/'.format(dataset_label, '_'.join(models_labels))}

    build_and_train(models_labels, dataset_label, train_params=train_params, save_folders=save_folders)

from keras.optimizers import Adam

from GAN_models import get_models
from Data_handlers import get_data_iterator, sample_results, save_models, make_folder
from GAN_template import GAN
from WGAN_template import WGAN


def set_default(model_params, train_params):
    if model_params is None:
        model_params = dict()
    if train_params is None:
        train_params = dict()

    default_model_params = {'model_type': 'fc',
                            'dataset': 'mnist',
                            'search_subfolders': False,
                            'data_type': None,
                            'load_full_dataset': True,
                            'latent_dim': 100,
                            'wgan': False,
                            'save_folders': None,
                            'imgs_resized_size': None,
                            'model_labels': ['fc']
                            }

    default_train_params = {'epochs': 30000,
                            'batch_size': 64,
                            'sample_interval': 1000,
                            'model_interval': 5000,
                            'wgan_clip': -1,
                            'label_smoothing': None,  # {'real': (.7, 1.3), 'gen': None},
                            'gen_to_disc_ratio': (1, 1)}

    for key, value in model_params.items():
        default_model_params[key] = value

    for key, value in train_params.items():
        default_train_params[key] = value

    return default_model_params, default_train_params


def build_and_train(model_params=None, train_params=None):
    """High level function for applying GAN on specific data sets.
    Takes as arguments two dictionaries:

    model_params:
        Dictionary with the parameters of the models for the generator and the discriminator and the save folders.
        The arguments of model_params are:

        - model_type:        (string or dictionary) if string must be one of 'fc' or 'cv' for denoting
                             the type of layers of the generator and the discriminator models.
                             'fc' stands for Fully connected (Dense) while 'cv' for convolutional (DCGAN).
                             If a dictionary, must have as keys 'generator' and 'discriminator'
                             with values the corresponding Kears models.
                             Default: 'fc'

        - dataset:           (string) can be either the name of an existing Keras dataset (e.g. mnist, cifar10 etc.)
                             or the (relative) path of a folder that contains the (equal sized) training images.
                             Default: 'mnist'

        - search_subfolders: (boolean) if True the model will search recursively
                             all the subfolders of the directory for images.
                             Default: False

        - data_type:         (string) with the data type (extension) of the training files (e.g. 'png')
                             if None it will read all the existing files (which must be cv2 compatible)
                             the number of the channels (e.g. if there is Alpha channel)
                             will be recognized automatically.
                             Default: None

        - load_full_dataset: (boolean) in case that data set is a folder, it defines if it will load all the images
                             or it will load the images paths and read the images at each iteration.
                             If data set fits in memory it is faster if the value is True.
                             Default: True

        - latent_dim:        (integer) the number of dimensions of the latent space to be used (size of the array).
                             Default: 100

        - wgan:              (boolean) if True Wasserstein training method will be applied
                             Default: False

        - save_folders:      (dictionary) with keys 'images_folder' and 'models_folder'
                             which have as values the directories where the results (images and models) will be saved
                             if None, the default paths will be used
                             (./result_images/dataset/ and ./result_models/dataset/).
                             Default: None

        - model_labels:      (list) of strings with different labels that can be used for determining the name
                             of the default saving folders and files. All the labels will be joined with '_'.
                             Default: None

    train_params:
        Dictionary with the training parameters
        The arguments of train_params are:

        - epochs:           (integer) number of epochs.
                            For each epoch it runs for as many repetitions defined at gen_to_disc_ratio.
                            Default: 30000

        - batch_size:       (integer) size of the batch of samples at each training iteration.
                            Default: 64

        - sample_interval:  (integer) denotes every how many epochs a number of generated samples will be saved.
                            Default: 1000

        - model_interval:   (integer) denotes every how many epochs the generator and discriminator models
                            will be saved.
                            Default: 5000

        - wgan_clip':       (float) if wgan is used, denotes the interval
                            of the discriminator weights clipping (-wgan_clip, wgan_clip).
                            Default: -1

        - label_smoothing': (dictionary or None) with keys 'real' and 'gen'
                            and values either a number or a tuple of two numbers or None.
                            - If None, no smoothing will take place for the specific category
                            - If a single number, the labels of this category will be equal to this number.
                            - If a tuple of two numbers, the labels will be uniformly sampled from this interval.
                            Default: None

        - gen_to_disc_ratio (tuple or function) if tuple of two integers the numbers will denote how many
                            times the generator and the discriminator will be trained at each epoch
                            if function it must take as argument the number of current epoch
                            and return a tuple as described above.
                            Default: (1, 1)
    """

    model_params, train_params = set_default(model_params, train_params)

    dataset_label = model_params['dataset']
    model_labels = model_params['model_labels']
    data_iterator, data_shape = get_data_iterator(dataset_label, model_params)
    model_params['data_shape'] = data_shape

    if model_params['save_folders'] is None:
        save_folders = {'images_folder': './result_images/{}/{}/'.format(model_params['dataset'],
                                                                         '_'.join(model_labels)),
                        'models_folder': './result_models/{}/{}/'.format(model_params['dataset'],
                                                                         '_'.join(model_labels))}
    else:
        save_folders = model_params['save_folders']

    for folder in save_folders.values():
        make_folder(folder)

    if isinstance(model_params['model_type'], str):
        models = get_models(model_params)
    else:
        models = model_params['model_type']

    if model_params['wgan']:
        gan = WGAN(model_params, models, save_folders, dataset_label, optimizer='RMSprop')
    else:
        gan = GAN(model_params, models, save_folders, dataset_label, optimizer=Adam(0.0002, 0.5))

    gan.train(data_iterator, sample_results, save_models, train_params)


def run():
    model_params = {'model_type': 'fc',
                    'dataset': 'mnist',
                    'search_subfolders': True,
                    'data_type': None,
                    'load_full_dataset': True,
                    'latent_dim': 100,
                    'wgan': False,
                    'save_folders': None,
                    'model_labels': None,
                    'imgs_resized_size': (64, 64)
                    }

    train_params = {'epochs': 100000,
                    'batch_size': 64,
                    'sample_interval': 1000,
                    'model_interval': 10000,
                    'wgan_clip': -1,
                    'label_smoothing': None,  # {'real': (.7, 1.3), 'gen': None},
                    'gen_to_disc_ratio': (1, 1)}

    model_labels = [model_params['model_type']]
    # append any labels to be used for the saving folders and files naming (e.g. model_labels.append('with_wgan')

    model_params['model_labels'] = model_labels

    build_and_train(model_params=model_params, train_params=train_params)


if __name__ == '__main__':
    # print(build_and_train.__doc__)
    run()

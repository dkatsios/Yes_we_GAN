# Yes_we_GAN
Small project on GANs. Experimenting with basic GAN types

The present project is an introductory attempt to experiment with different kinds of GANs architectures and datasets. It is based on various papers, tutorials, blog posts and github repositories, some of which are mentioned as sources at the code files.
The main idea is to build a very simple &quot;framework&quot; for applying different types of GANs on different datasets easily.
In order to do so, the parts of the algorithm have been separated based on their functionality, resulting to four (4) main files: GAN\_models.py, xGAN\_template.py, Data\_handlers.py and GAN\_builder.py.

___GAN\_models.py___
This file contains the code for building the generator and discriminator models. For every type of model, there are two functions:

- x\_gan\_generator(dimensions): returns an x type model for the generator based on the given dimensions
- x\_gan\_discriminator(dimensions, final\_act): returns an x type model for the discriminator based on the given dimensions and with _final\_act_ as activation function for the output layer. This is needed in the case of wasserstein GANs, where the activation function should be _linear_ instead of _sigmoid_.

The currently supported types of models are simple fully connected (s) and convolutional (cv). The file also contains the get\_models(label, dimensions) function which takes as inputs a label in the form of string and the dimensions dictionary and returns a dictionary with the two models.

___xGAN\_template.py___
This file contains the xGAN class which is a class that determines the functionality of the GAN. The class is initialized with these arguments:
- dimensions: the shape of the data tensor (usually an image e.g. 32x32x3)
- models: the dictionary with the generator and the discriminator models
- save\_folders: the directories for saving generated images and the model weights
- dataset: an iterator that generates real examples in batches
- ptimizer: the optimizer used for the training of the models

The main methods of the class are the \_\_init\_\_() and the train() methods:
- \_\_init\_\_(): initializes the attributes of the object, the models of generator and discriminator and the combined GAN model. It also defines the input and output tensors and finally compiles the models based on the given optimizer.
- train(): this method is responsible for the GAN training, it produces the real and generated data and trains the combined model and the generator sequentially possibly with different ratio. This method also saves sampled generated data and the weights of the models.

___Data\_handlers.py___
The main function of this file is get\_data\_iterator(dataset, load\_full) which takes as input:

- dataset: either the name of Keras contained datasets (e.g. mnist) or the directory containing the training data
- load\_full: in the case that dataset is a directory&#39;s path, this boolean parameter determines whether it will load the all the data at a tensor or if the iterator will read the data on the spot

The function return an iterator of training data that generates in batches.

___GAN\_builder.py___
- The main function of this file is build\_and\_train() which takes as input arguments:
- models\_label: a string determining the type of models and technique used for the GAN
- dataset\_label: the label of the training dataset
- im\_dim: the dimensions of the data
- train\_params: a dictionary with the training hyperparameters

### Supported GAN types
For the moment only a very limited number of architectures and methods are supported. However, it is very easy to construct a new neural network model as a generator and discriminator and use it via the &quot;framework&quot;. For GAN types, for the moment only the simple vanilla GAN and the wasserstein GANs are supported, but conditional GANs will be added soon. Regarding the training functionalities, a different ratio of generator / discriminator training circles is supported, important mainly in the case of WGANs, where the discriminator may need to be trained more times that the generator per repetition. Also, label smoothing is supported, mainly for simple GANs, where the labels of the real and generated data can be other than 1 and 0 respectively. One can change the one or both labels to be equal to different number or to be sampled for a uniform distribution between two numbers.

### Datasets

The datasets used for training the possible combinations of the supported GAN types are:

- MNIST: The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image [1].
- Flags: a subset of emoji symbols downloaded from [2]. There are 259 flags from different countries in circle shape of sizes 32x32, 64x64 and 128x128. The models were trained on the first two sizes because of the limited time and resources. It is worthwhile to mention that this is not an official dataset for (GANs) training and thus the results are not guaranteed to be of high quality. However it is interesting to apply methods like these on new datasets different than the classic and overused mnist-kind ones.

The only preprocessing that took place was the normalization of the pixel RGBA values to [-1, 1]. No data augmentation was used.

### Results
MNIST
- Fully Connected GAN:
- Fully Connected WGAN:
- Convolutional GAN:
- Convolutional WGAN:

FLAGS
- Fully Connected GAN:
- Fully Connected WGAN:
- Convolutional GAN:
- Convolutional WGAN:

### Conclusions

A simple GANs &quot;framework&quot; was developed for easier implementation of standard models and training techniques on different datasets.

[1] LeCun, Yann, Corinna Cortes, and C. J. Burges. &quot;MNIST handwritten digit database.&quot; AT&amp;T Labs [Online]. Available: http://yann. lecun. com/exdb/mnist 2 (2010).

[2] EmojiOne™: emojione.com/download



# Generative Deep Learning

## Contents
- [Running apps](#running-apps)
  - [Multi Layer Perceptron MLP](#multi-layer-perceptron-mlp)
  - [Convolutional Neural Network CNN](#convolutional-neural-network-cnn)
  - [Autoencoder AE](#autoencoder-ae)
  - [Variational Autoencoder VAE](#variational-autoencoder-vae)
  - [Generative Adversarial Network GAN](#generative-adversarial-network-gan)
- [Repo Structure](#repo-structure)
- [Getting started](#getting-started)
  - [Install libraries](#install-libraries)
  - [Conda environment set-up](#conda-environment-set-up)
  - [Start notebook server](#start-notebook-server)
- [Install data and other supporting apps](#install-data-and-other-supporting-apps)
  - [CIFAR-10](#cifar-10)
  - [MNIST](#mnist)
  - [CelebA](#celeba)
  - [Music3 data and software](#music3-data-and-software)
- [Future](#future)

## Running apps

Applications should be run from `src/`, relative to the `src/generator` module.

After setting up the environment and retrieving the requisite datasets, run the application with `-h` to list usage.

Trained models for some of these applications are provided under `saved_models`, where noted below.

### Multi Layer Perceptron MLP

No data needs to be retrieved for this simple application (not generative).

`python cifar10_mlp.py`

### Convolutional Neural Network CNN

No data needs to be retrieved for this simple application (not generative).

`python cifar10_cnn.py`

### Autoencoder AE

No data needs to be retrieved for this simple application (not generative).

`python mnist_ae_train.py`

### Variational Autoencoder VAE
The VAEs can be run on MNIST hand-writing data as follows:
 - `python mnist_vae_train.py`
 - `python mnist_vae_analysis.py` - can run immediately by copying model from `saved_models/mnist_vae/` to `src/run/vae/0001_digits/`
 
 Install the celeb-A data (see below) to run the following:
 - `python celeba_vae_train.py` - training takes hours on a consumer-grade GPU
 - `python celeba_vae_analysis.py` - can run immediately by copying model from `saved_models/celeb_vae/` to `src/run/vae/0001_faces/`

### Generative Adversarial Network GAN
Coming Soon!

## Repo Structure

This repository is structured as follows:
 - `src/`:  **toy applications** in the root, and all relevant source code under `generator`
 - `src/generator`:  models and useful functions that are sourced by the examples
 - `src/run`:  created automatically by the toy examples, stores output from the generative models 
 - `saved_models`:  weights saved from models trained by apps under `src`. These can be loaded and used immediately for predictions without training if the user has knowledge of how to do that.
 - `generative_env.yml`: a tested conda environment; note, `LD_LIBRARY_PATH` variable must also be **exported**, see below

## Getting started

### Install libraries

There are a few different approaches to running this code. One is to
create a Google colab notebook and import the data/code there. Another
is to set up docker and run a GPU-enabled container. Below are
instructions for a third option, set up a python environment. These
instructions have been tested on Ubuntu.

To get started, first install the required libraries inside a virtual environment:

```
# install nvidia drivers if you haven't already:
#https://www.nvidia.com/Download/index.aspx

# make a tensorflow environment that works with
# 11th Gen Intel(R) Core(TM) i9-11900H @ 2.50GHz   2.50 GHz
# check your card:
nvidia-smi --query-gpu=gpu_name --format=csv|tail -n 1
# NVIDIA GeForce RTX 3050 Ti Laptop GPU


# install mamba for faster package management:
# sometimes you have to repeat a mamba command, its still faster than conda
conda install -n base conda-forge::mamba
```
###  Conda environment set-up

Create a virtual environment using mamba (because it's faster than conda).

```
mamba create -n generative tensorflow-gpu -c conda-forge
conda activate generative
# tested under:
# python v3.9.13, tensorflow v 2.6.2; you'll lget the right version of libcusolver.so (v11) for cudatoolkit=11.7
# 
# CAVEAT: tensorflow libraries and nvidia drivers have some strange inter-dependencies and bugs that make package management dodgey, at best, especially in earlier chipsets. In my experience, these are becoming more stable with evolving tensorflow/keras combinations, and using the latest hardware/drivers/libs, suchas offered by Google colab, go a long way towards solving this problem. However, the following notes include hints for overcoming problems that may be caused by GPU chipsets/drivers that may not be the most current.

pip install tf-explain

# #############
# BEGIN: UPGRADE (or downgrade) TF:
# If you wantt o upgrade tf, do at your own risk:
# install/uninstall/install some tf/cuda packages to get the right version combination

# uninstall tensorflow so it's linked to the version your cudakit needs later on
pip uninstall tensorflow

# xxx this leaves behind libcusolver11.so.11 somehow?
conda uninstall cudatoolkit # mamba didn't work for me

# install tool to query your nvidia toolkit version
mamba install cuda-nvcc -c nvidia
nvcc --version

# Assuming nvcc version is 11.7: will bring correct cudnn, and libcusolver.so.11:
# WARNING: 2.5.0 has a broken libcusolver
#   If you install cudatoolkit any other way for 2.5.0, libcusolver.so.10 will be installed when you need so.11, and you'll get errors
mamba install cudatoolkit=11.7 

# IF your cuda drivers need an explicit tf version, uncomment and modify the code below:
#TENSORFLOW_VERSION="==2.3" 
# ELSE use htis:
TENSORFLOW_VERSION="" # use this to explicitly set version if required by your cuda drivers
pip install tensorflow-gpu${TENSORFLOW_VERSION}
# ensure you still have the right version of libcusolver
#ls ~/miniconda3/envs/generative/lib/libcusolver*
# or 
ls $CONDA_PREFIX/lib/libcusolver*
# END: UPGRADE (or downgrade) TF
# ######################

# make sure tf can find your cuda libraries
pushd $CONDA_PREFIX
mkdir -p ./etc/conda/activate.d
mkdir -p ./etc/conda/deactivate.d
echo -e "#\!/bin/sh\nexport SAVE_PREVIOUS_LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}\nexport LD_LIBRARY_PATH=${CONDA_PREFIX}/lib" > ./etc/conda/activate.d/env_vars.sh
echo -e "#\!/bin/sh\nexport LD_LIBRARY_PATH=\${SAVE_PREVIOUS_LD_LIBRARY_PATH}"
popd
# or simply:
# conda env config vars set LD_LIBRARY_PATH=~/miniconda3/envs/generative/lib

# pick up the environmental variable you just set
conda deactivate
conda activate generative

mamba install --file requirements.txt -c conda-forge -c esri
# add in latest from keras-contrib to pick up new InstanceNormalization layer
pip install git+https://www.github.com/keras-team/keras-contrib.git

### test:
python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
# ignore NUMA node warnings, they're harmless, see: https://forums.developer.nvidia.com/t/numa-error-running-tensorflow-on-jetson-tx2/56119/2
# I think this happens if GPU is number '0'
# All the libs should load
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# ignore NUMA node warning, see above

# CAVEAT:
# There's a weird bug in tensorflow package management that causes keras to be installed twice; 
# It might be from installation of keras-applications in requirements.txt, not sure
# if the tests above fail, try pip uninstall'ing keras by uncomment'ing below; and they should work again:
#pip uninstall keras

# save your environment so you can version control and/or use it on other machines, if desired
conda env export --file generative_env.yml
# CAVEAT - LD_LIBRARY_PATH must be exported for tf to work; but the yml above only 'sets' it.
```

### Start notebook server
```
jupyter notebook
```
You can monitor your nvidia process with the following:
```
nvidia-smi dmon
```

## Install data and other supporting apps

Install data to `./src/data` to run with the provided apps

### CIFAR-10

(CIFAR-10)[https://www.cs.toronto.edu/~kriz/cifar.html] has 80 million tiny images, each labeled with one of 10 classes: 
`'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'`

Nothing extra needs to be done to retrieve these data; The model loads them directly during runtime with: `cifar10.load_data()`

### mnist

(MNIST)[http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html] has 60k examples of labeled, hand-written digits.

Nothing extra needs to be done to retrieve these data; The model loads them directly during runtime with: `load_mnist()`.

### CelebA

(CelebA)[http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html] has 200K+ images of celebrity faces labeled with one of 40 classes. Read more here: 

This dataset must be downloaded to a directory structure common for CNN training. The following steps detail how to:

 * download the data from celeba (200,000+ faces, ) and extract to `generator/data/celeba/`. This will result in 200k+ jpeg's under `data/celeba/img_align_celeba`, named `000001.jpg`, `000002.jpg`, etc.
 * Set-up the labels for latent-space arithmatic

Detailed steps: 
 1. Get a google account if you don't have one.
 2. Go to the celebA website: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
 3. Select the Aligned&Cropped Images. This will take you to the Google Cloud Drive site.: https://drive.google.com/open?id=0B7EVK8r0v71pWEZsZE9oNnFzTm8
 4. Sign in with your Google account
 5. Download: 
   a. Colab Notebooks > GAN > CelebA > Anno > `list_attr_celeba.txt`
   b. Colab Notebooks > GAN > CelebA > Img > `img_align_celeba.zip`
 6. Unzip img_align_celeba.zip to `generator/data/celeb`. This will create `img_align_celeba/*jpg`
 7. Move the list_attr_celeba.txt to `generator/data/celeb/list_attr_celeba.csv`.
 8. Delete the first line with count of the number of lines in the file (202599).
 9. Prepend the header line that start with, "5_o_Clock_Shadow" with "image_id,...".

Something like this:
```
cat list_attr_celeba.txt | tr -s ' '| sed 's/ /,/g' |tail +2|sed 's/5_o_Clock_Shadow/image_id,5_o_Clock_Shadow/g'> src/data/celeb/list_attr_celeba.csv
```

or get it from kaggle: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset?resource=download


**Note**:
To use a batch_size of 32 (recommended), the number of files should be divisible by 8, but this dataset has len(filenames) % 8 = 7, so remove 7 files; something like this:

```
cd data/celeb/class_1
mv 202599.jpg 202598.jpg 202597.jpg 202596.jpg 202595.jpg 202594.jpg 202593.jpg ../../celeb-extras
```

If you don't ensure the number of files is a multiple of the batch-size, you may get an error similar to the following with tensorflow v2:
```
ValueError: operands could not be broadcast together with shapes (32,) (14,) (32,)
W tensorflow/core/kernels/data/generator_dataset_op.cc:108] Error occurred when finalizing GeneratorDataset iterator: FAILED_PRECONDITION: Python interpreter state is not initialized. The process may be terminated.

```

### Music3 data and software

Install music3 data for the music generator. These commands only works for ubuntu:
```
sudo add-apt-repository ppa:mscore-ubuntu/mscore-stable
sudo apt-get update
sudo apt-get install musescore
# mscore v2 is installed, but the GAN looks for mscore3:
sudo ln -s /usr/bin/mscore /usr/bin/mscore3
```
You can also get mscore 3 at https://musescore.org for other operating systems

More music:
 * Bach midis here: http://www.jsbach.net/midi/midi_solo_cello.html
 * Chorales here: https://github.com/czhuang/JSB-Chorales-dataset
 * More data here: https://github.com/Skuldur/Classical-Piano-Composer
 
Read more here:
 * https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5

# Future
Following are future features:
* GAN
* args to celeb, gan
* random_seed to models
* history to models


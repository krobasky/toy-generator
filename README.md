# Generative Deep Learning

## Structure

This repository is structured as follows:

`generator`: source code and classes for toy programs and models
`nb`:  jupyter notebooks with examples
`data`:  relevant data sources 
`run`:  stores output from the generative models 
`generator/utils`:  useful functions that are sourced by the main notebooks

## Getting started

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

###
# BEGIN [1] Conda environment set-up
###

mamba create -n generative tensorflow-gpu -c conda-forge
conda activate generative
# tested under:
# python v3.9.13, tensorflow v 2.6.2; you'll lget the right version of libcusolver.so (v11) for cudatoolkit=11.7
# 
# CAVEAT: tensorflow libraries and nvidia drivers have some strange inter-dependencies and bugs that make package management dodgey, at best, especially in earlier chipsets. In my experience, these are becoming more stable with evolving tensorflow/keras combinations, and using the latest hardware/drivers/libs, suchas offered by Google colab, go a long way towards solving this problem. However, the following notes include hints for overcoming problems that may be caused by GPU chipsets/drivers that may not be the most current.

pip install tf-explain

##############
# BEGIN [2]: UPGRADE (or downgrade) TF:
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
# END [2]: UPGRADE (or downgrade) TF
#######################

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

###
# END [1] Conda environment set-up
###

####
# Install data and other apps
###

# install music3 for the music generator; this command only works for ubuntu:
sudo add-apt-repository ppa:mscore-ubuntu/mscore-stable
sudo apt-get update
sudo apt-get install musescore
# mscore v2 is installed, but the GAN looks for mscore3:
sudo ln -s /usr/bin/mscore /usr/bin/mscore3
# https://musescore.org for other operating systems

# download Bach midis here:
# http://www.jsbach.net/midi/midi_solo_cello.html
# and chorales here:
# https://github.com/czhuang/JSB-Chorales-dataset
# read more here:
# https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5
# and find more data here:
# https://github.com/Skuldur/Classical-Piano-Composer
# Celebrity faces for data/celeb:
# http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

# start your notebook server
#LD_LIBRARY_PATH=~/miniconda3/envs/generative/lib 
jupyter notebook

# you can monitor your nvidia process with the following:
nvidia-smi dmon
 
```
 




#!/usr/bin/env python

# # VAE Training - Faces dataset

import os
from glob import glob
import numpy as np

from models.VAE import VariationalAutoencoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator

'''
The following steps detail how to:
A. download the data from celeba (200,000+ faces, ) and extract to TOY_ROOT/data/celeba/
This will result in 200k+ jpeg's under data/celeba/img_align_celeba, named '000001.jpg', '000002.jpg', etc.
B. Set-up the labels for latent-space arithmatic

 1. Get a google account if you don't have one.
 2. Go to the celebA website: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
 3. Select the Aligned&Cropped Images. This will take you to the Google Cloud Drive site.: https://drive.google.com/open?id=0B7EVK8r0v71pWEZsZE9oNnFzTm8
 4. Sign in with your Google account
 5. Download
 6. Colab Notebooks > GAN > CelebA > Anno > list_attr_celeba.txt
 7. Colab Notebooks > GAN > CelebA > Img > img_align_celeba.zip
 8. Unzip img_align_celeba.zip to path/to/GDL_Code/data/celeb.
   This will create img_align_celeba/*jpg
 9. Move the list_attr_celeba.txt to ${path_to_toy-generator/data/celeb/list_attr_celeba.csv.
 10. Delete the first line with count of the number of lines in the file (202599).
 11. Prepend the header line that start with, "5_o_Clock_Shadow" with "image_id,...".

or get it from kaggle: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset?resource=download

With data in-place, the following code can be run as-is to build the VAE model.
'''

# run params
section = 'vae'
run_id = '0001'
data_name = 'faces'
RUN_FOLDER = 'run/{}/'.format(section)
RUN_FOLDER += '_'.join([run_id, data_name])
if not os.path.exists(RUN_FOLDER):
    os.makedirs(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
    os.mkdir(os.path.join(RUN_FOLDER, 'images'))
    os.mkdir(os.path.join(RUN_FOLDER, 'weights'))
mode =  'build' #'load' #
DATA_FOLDER = './data/celeb/'

# ## data
INPUT_DIM = (128,128,3)
BATCH_SIZE = 32
filenames = np.array(glob(os.path.join(DATA_FOLDER, '*/*.jpg')))
NUM_IMAGES = len(filenames)
data_gen = ImageDataGenerator(rescale=1./255)
data_flow = data_gen.flow_from_directory(DATA_FOLDER
                                         , target_size = INPUT_DIM[:2]
                                         , batch_size = BATCH_SIZE
                                         , shuffle = True
                                         , class_mode = 'input'
                                         , subset = "training"
                                            )
# ## architecture
from models.VAE import VariationalAutoencoder
vae = VariationalAutoencoder(
                input_dim = INPUT_DIM 
                , encoder_conv_filters=[32,64,64, 64]
                , encoder_conv_kernel_size=[3,3,3,3]
                , encoder_conv_strides=[2,2,2,2]
                , decoder_conv_t_filters=[64,64,32,3]
                , decoder_conv_t_kernel_size=[3,3,3,3]
                , decoder_conv_t_strides=[2,2,2,2]
                , z_dim=200
                , use_batch_norm=True
                , use_dropout=True
                , r_loss_factor = 10000
                )
if mode == 'build':
    vae.save(RUN_FOLDER)
else:
    vae.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))
#vae.encoder.summary()
#vae.decoder.summary()

# ## training
LEARNING_RATE = 0.0005
EPOCHS = 200
PRINT_EVERY_N_BATCHES = 100
INITIAL_EPOCH = 0
vae.compile(LEARNING_RATE)
vae.train_with_generator(     
    data_flow
    , epochs = EPOCHS
    , steps_per_epoch = NUM_IMAGES / BATCH_SIZE
    , run_folder = RUN_FOLDER
    , print_every_n_batches = PRINT_EVERY_N_BATCHES
    , initial_epoch = INITIAL_EPOCH
)

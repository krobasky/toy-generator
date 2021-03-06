#!/usr/bin/env python
'''
usage: python ./celeba_vae_train.py
  Trains a VAE on images and saves the model.

  Inputs:
    data/celeb/*/*.jpg
  Outputs:
    run/vae/0001/faces/weights/weights.h5
'''
import os
from glob import glob
import numpy as np

from generator.model.VAE import VariationalAutoencoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
extra_images = NUM_IMAGES % BATCH_SIZE
if extra_images != 0:
    print(f"ERROR: number of images[{NUM_IMAGES}] is not a multiple of batch size[{BATCH_SIZE}]; Please remove [{extra_images}] images from folder=[{DATA_FOLDER}]")
    exit()

data_gen = ImageDataGenerator(rescale=1./255)
data_flow = data_gen.flow_from_directory(DATA_FOLDER
                                         , target_size = INPUT_DIM[:2]
                                         , batch_size = BATCH_SIZE
                                         , shuffle = True
                                         , class_mode = 'input'
                                         , subset = "training"
                                            )
# ## architecture
from generator.model.VAE import VariationalAutoencoder
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

vae.encoder.summary()
vae.decoder.summary()

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

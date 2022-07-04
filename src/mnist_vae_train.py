#!/usr/bin/env python
from generator.argparse import p
opts=p(
"""Demonstrates VAE training.
  Inputs: 
    Downloads MNIST from internet
  Outputs: (STDOUT)
    - model summary
    - verbose training epochs
    - model loss and accuracy in final line"""
    , epochs=200
    , learning_rate=0.0005
    , batch_size=32
    , input_shape=(28,28,1)
    , scale_value=255.0
    , filter=[32,64,64, 64]
    , kernel_size=[3,3,3,3]
    , stride=[1,2,2,1]
    , z_dim=2
    , d_filter=[64,64,32,1]
    , d_kernel_size=[3,3,3,3]
    , d_stride=[1,2,2,1]
    , r_loss_factor=1000
    , print_every_n_batches=100
    , initial_epoch=0 
)

INPUT_SHAPE= opts.input_shape
SCALE_VALUE= opts.scale_value
EPOCHS= opts.epochs
VERBOSE=1
if opts.quiet == True:
    VERBOSE=2
BATCH_SIZE=opts.batch_size
LEARNING_RATE=opts.learning_rate

FILTER=opts.filter
KERNEL_SIZE=opts.kernel_size
STRIDE=opts.stride

D_FILTER=opts.d_filter
D_KERNEL_SIZE=opts.d_kernel_size
D_STRIDE=opts.d_stride
R_LOSS_FACTOR = opts.r_loss_factor
PRINT_EVERY_N_BATCHE = opts.print_every_n_batches
INITIAL_EPOCH = opts.initial_epoch

Z_DIM = opts.z_dim
PRINT_EVERY_N_BATCHES=opts.print_every_n_batches

# # VAE Training
# ## imports
import os
from generator.model.VAE import VariationalAutoencoder
from generator.loaders import load_mnist

# run params
SECTION = 'vae'
RUN_ID = '0002'
DATA_NAME = 'digits'
RUN_FOLDER = 'run/{}/'.format(SECTION)
RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])
if not os.path.exists(RUN_FOLDER):
    os.makedirs(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
    os.mkdir(os.path.join(RUN_FOLDER, 'images'))
    os.mkdir(os.path.join(RUN_FOLDER, 'weights'))
mode =  'build' #'load' #

# ## data
(x_train, y_train), (x_test, y_test) = load_mnist()
# ## architecture
vae = VariationalAutoencoder(
    input_dim = INPUT_SHAPE
    , encoder_conv_filters = FILTER
    , encoder_conv_kernel_size = KERNEL_SIZE
    , encoder_conv_strides = STRIDE
    , decoder_conv_t_filters = D_FILTER
    , decoder_conv_t_kernel_size = D_KERNEL_SIZE
    , decoder_conv_t_strides = D_STRIDE
    , z_dim = Z_DIM
    , r_loss_factor = R_LOSS_FACTOR
)
if mode == 'build':
    vae.save(RUN_FOLDER)
else:
    vae.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))

vae.encoder.summary()
vae.decoder.summary()

# ## train and save
vae.compile(LEARNING_RATE)
vae.train(     
    x_train
    , batch_size = BATCH_SIZE
    , epochs = EPOCHS
    , run_folder = RUN_FOLDER
    , print_every_n_batches = PRINT_EVERY_N_BATCHES
    , initial_epoch = INITIAL_EPOCH
    , verbose = VERBOSE
)

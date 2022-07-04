#!/usr/bin/env python
from generator.argparse import p
opts=p(
"""MNIST Encoder.
  Inputs: 
    Downloads MNIST from internet
  Outputs: (STDOUT)
    - model summary
    - verbose training epochs
    - model loss and accuracy in final line"""
    , epochs=10
    , learning_rate=0.0005
    , batch_size=32
    , input_shape=(32,32,3)
    , num_classes=10
    , scale_value=255.0
    , filter=[64,64,32,1]
    , kernel_size=[3,3,3,3]
    , stride=[1,2,2,1]
    , z_dim = 200
)

INPUT_SHAPE= opts.input_shape
SCALE_VALUE= opts.scale_value
EPOCHS= opts.epochs
VERBOSE=True
if opts.quiet == True:
    VERBOSE=False
BATCH_SIZE=opts.batch_size
LEARNING_RATE=opts.learning_rate

FILTER=opts.filter
KERNEL_SIZE=opts.kernel_size
STRIDE=opts.stride
Z_DIM=opts.z_dim

import os
from generator.loaders import load_mnist
from generator.model.AE import Autoencoder

# ## Set parameters
# run params
MODEL_TYPE = 'vae'
RUN_ID = '0001'
DATA_NAME = 'digits'
RUN_FOLDER = 'run/{}/'.format(MODEL_TYPE)
RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])
if not os.path.exists(RUN_FOLDER):
    os.makedirs(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
    os.mkdir(os.path.join(RUN_FOLDER, 'images'))
    os.mkdir(os.path.join(RUN_FOLDER, 'weights'))
MODE =  'build' #'load' #

# ## Load the data
(x_train, y_train), (x_test, y_test) = load_mnist()

# ## Define the structure of the neural network
AE = Autoencoder(
    input_dim = (28,28,1)
    , encoder_conv_filters = FILTER
    , encoder_conv_kernel_size = KERNEL_SIZE
    , encoder_conv_strides = STRIDE
    , decoder_conv_t_filters = FILTER
    , decoder_conv_t_kernel_size = KERNEL_SIZE
    , decoder_conv_t_strides = STRIDE
    , z_dim = Z_DIM
)

if MODE == 'build':
    AE.save(RUN_FOLDER)
else:
    AE.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))

AE.encoder.summary()
AE.decoder.summary()

# ## Train the autoencoder
INITIAL_EPOCH = 0
AE.compile(LEARNING_RATE)
AE.train(     
    x_train[:1000]
    , batch_size = BATCH_SIZE
    , epochs = EPOCHS
    , run_folder = RUN_FOLDER
    , initial_epoch = INITIAL_EPOCH
    , verbose = VERBOSE
)


#!/usr/bin/env python
from generator.argparse import p
opts=p(
"""Demonstrates CNN improves performance over MLP models.
  Inputs: 
    Downloads cifar from internet
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
    , filter=(32,32,64,64)
    , kernel_size=(3,3,3,3)
    , stride=(1,2,1,2)
)

INPUT_SHAPE= opts.input_shape
SCALE_VALUE= opts.scale_value
EPOCHS= opts.epochs
VERBOSE=True
if opts.quiet == True:
    VERBOSE=False
BATCH_SIZE=opts.batch_size
LEARNING_RATE=opts.learning_rate

FILTER=(32,32,64,64)
KERNEL_SIZE=(3,3,3,3)
STRIDE=(1,2,1,2)

import numpy as np

from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, BatchNormalization, LeakyReLU, Dropout, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K 

from tensorflow.keras.datasets import cifar10
# # data
CLASSES = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
NUM_CLASSES = len(CLASSES)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / SCALE_VALUE
x_test = x_test.astype('float32') / SCALE_VALUE
y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)
# x_train[54, 12, 13, 1] 

# # architecture
'''
input_layer = Input(shape=INPUT_SHAPE)
conv_layer_1 = Conv2D(
    filters = 10
    , kernel_size = (4,4)
    , strides = 2
    , padding = 'same'
    )(input_layer)
conv_layer_2 = Conv2D(
    filters = 20
    , kernel_size = (3,3)
    , strides = 2
    , padding = 'same'
    )(conv_layer_1)
flatten_layer = Flatten()(conv_layer_2)
output_layer = Dense(units=10, activation = 'softmax')(flatten_layer)
model = Model(input_layer, output_layer)
model.summary()
'''

             
input_layer = Input(INPUT_SHAPE)

i=0
# number of parameters = 
# (kernel-x * kernel-y + bias-term) * filters = 
# (3 * 3 * 3 + 1) * 32 = 896 parameters
# output-shape: no-observations, height/strides, width/strides, filters
#   None, 32, 32, 32
x = Conv2D(filters = FILTER[i], kernel_size = KERNEL_SIZE[i], strides = STRIDE[i], padding = 'same')(input_layer)
# (output-channels * 2-trainable-parameters * 2-derived-parameters) =
# 32 * 2 * 2 = 128 parameters(64 trainable)
# output-shape: same as input
x = BatchNormalization()(x)
# no paremeters, pass through
x = LeakyReLU()(x)
i=i+1

# (kernel-x * kernel-y * previous-channels + bias-term) * filters = 
# (3 * 3 * 32 + 1) * 32 = 9248 parameters
# output-shape: no-observations, height/strides, width/strides, filters
#   None, 16, 16, 32
x = Conv2D(filters = FILTER[i], kernel_size = KERNEL_SIZE[i], strides = STRIDE[i], padding = 'same')(x)
# (output-channels * 2-trainable-parameters * 2-derived-parameters) =
# (32 * 2 * 2) = 128 parameters (64 trainable) 
x = BatchNormalization()(x)
# no paremeters, pass through
x = LeakyReLU()(x)
i=i+1

# (kernel-x * kernel-y * previous-channels + bias-term) * filters = 
# (3 * 3 * 32 + 1) * 64 = 18496 parameters
# output-shape: no-observations, height/strides, width/strides, filters
#   None, 16, 16, 64
x = Conv2D(filters = FILTER[i], kernel_size = KERNEL_SIZE[i], strides = STRIDE[i], padding = 'same')(x)
# (output-channels * 2-trainable-parameters * 2-derived-parameters) =
# (64 * 2 * 2) = 256 parameters (128 trainable) 
x = BatchNormalization()(x)
# no paremeters, pass through
x = LeakyReLU()(x)
i=i+1

# (kernel-x * kernel-y * previous-channels + bias-term) * filters = 
# (3 * 3 * 64 + 1) * 64 =  36,928 parameters
# output-shape: no-observations, height/strides, width/strides, filters
#   None, 8, 8, 64
x = Conv2D(filters = FILTER[i], kernel_size = KERNEL_SIZE[i], strides = STRIDE[i], padding = 'same')(x)
# (output-channels * 2-trainable-parameters * 2-derived-parameters) =
# (64 * 2 * 2) = 256 parameters (128 trainable) 
x = BatchNormalization()(x)
# no paremeters, pass through
x = LeakyReLU()(x)

# no paremeters, pass through
# output-shape: no-observations, height * width/strides * channels
#   None, 8*8*64=
#   None, 4096
x = Flatten()(x)

# (input + bias-term) * output =
# (4096 + 1) * 128 = 524416 parameters
# output-shape: no-observations, output
#  None, 128
x = Dense(128)(x)
# (output-channels * 2-trainable-parameters * 2-derived-parameters) =
# (128 * 2 * 2) = 512 parameters (256 trainable) 
x = BatchNormalization()(x)
# no paremeters, pass through
x = LeakyReLU()(x)
# no paremeters, pass through
x = Dropout(rate = 0.5)(x)

# (input + bias-term) * output =in
# (128 + 1) * 10 = 1290 parameters
# output-shape: no-observations, output
#  None, 10
x = Dense(NUM_CLASSES)(x)
# no paremeters, pass through
output_layer = Activation('softmax')(x)

model = Model(input_layer, output_layer)
model.summary()

# # train
opt = Adam(learning_rate=LEARNING_RATE)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(x_train
          , y_train
          , batch_size=BATCH_SIZE
          , epochs=EPOCHS
          , shuffle=True
          , validation_data = (x_test, y_test)
          , verbose = VERBOSE
          )

#model.layers[6].get_weights() # not sure what this is for

# # analysis
model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)


'''
preds = model.predict(x_test)
preds_single = CLASSES[np.argmax(preds, axis = -1)]
actual_single = CLASSES[np.argmax(y_test, axis = -1)]

# better for notebooks
import matplotlib.pyplot as plt

n_to_show = 10
indices = np.random.choice(range(len(x_test)), n_to_show)

fig = plt.figure(figsize=(15, 3))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i, idx in enumerate(indices):
    img = x_test[idx]
    ax = fig.add_subplot(1, n_to_show, i+1)
    ax.axis('off')
    ax.text(0.5, -0.35, 'pred = ' + str(preds_single[idx]), fontsize=10, ha='center', transform=ax.transAxes) 
    ax.text(0.5, -0.7, 'act = ' + str(actual_single[idx]), fontsize=10, ha='center', transform=ax.transAxes)
    ax.imshow(img)
'''




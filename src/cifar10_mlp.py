#!/usr/bin/env python
from generator.argparse import p
opts=p(
"""Demonstrates how much room for improvement there is for image classification when using only a simple MLP.
  Inputs: 
    Downloads cifar from internet
  Outputs: (STDOUT)
    - model summary
    - verbose training epochs
    - model loss and accuracy in final line"""
    , epochs=3
    , learning_rate=0.0005
    , batch_size=32
    , input_shape=(32,32,3)
    , num_classes=10
    , scale_value=255.0
)

NUM_CLASSES = opts.num_classes
INPUT_SHAPE= opts.input_shape
SCALE_VALUE= opts.scale_value
EPOCHS= opts.epochs
VERBOSE=1
if opts.quiet == True:
    VERBOSE=2
BATCH_SIZE=opts.batch_size
LEARNING_RATE=opts.learning_rate

# # imports
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10

# # data 

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / SCALE_VALUE
x_test = x_test.astype('float32') / SCALE_VALUE
y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)
#x_train[54, 12, 13, 1] 

# # architecture
input_layer = Input(INPUT_SHAPE)
x = Flatten()(input_layer)
# after 5 trials with 3 epochs (to save time), the difference between 1-layer with more weights vs 2-layers is not statistically significant in a t-test
#x = Dense(1531, activation = 'relu')(x)
x = Dense(350, activation = 'relu')(x)
#x = Dense(200, activation = 'relu')(x)
#x = Dense(150, activation = 'relu')(x)
output_layer = Dense(NUM_CLASSES, activation = 'softmax')(x)
model = Model(input_layer, output_layer)
model.summary()

# # train
#opt = Adam(learning_rate=0.0005)
opt = Adam(learning_rate=LEARNING_RATE)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(x_train
          , y_train
          , batch_size=BATCH_SIZE
          , epochs=EPOCHS
          , shuffle=True
          , verbose=VERBOSE
          )

# # analysis
model.evaluate(x_test, y_test)

'''
# works better in a notebook
CLASSES = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
preds = model.predict(x_test)
preds_single = CLASSES[np.argmax(preds, axis = -1)]
actual_single = CLASSES[np.argmax(y_test, axis = -1)]

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

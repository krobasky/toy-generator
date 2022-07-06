#!/usr/bin/env python
# #####
# ## initialize
# #####
# ## imports
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import norm
import pandas as pd

from generator.model.VAE import VariationalAutoencoder
from generator.loaders import load_model, ImageLabelLoader
from generator.viz import demo_linalg_norm, vector_distribution, keypress_next_images, compare_images, add_vector_to_images, morph_faces
from generator.latent import label2z
# run params
section = 'vae'
run_id = '0001'
data_name = 'faces'
RUN_FOLDER = 'run/{}/'.format(section)
RUN_FOLDER += '_'.join([run_id, data_name])
DATA_FOLDER = './data/celeb/'
IMAGE_FOLDER = './data/celeb/img_align_celeba/'
# ## data
INPUT_DIM = (128,128,3)
SCALE_VALUE=255
filepath=os.path.join(DATA_FOLDER, 'list_attr_celeba.csv')
files_by_attribute = pd.read_csv(filepath)
files_by_attribute=files_by_attribute.drop(columns=["Unnamed: 41"]) # weirdly appended column

imageLoader = ImageLabelLoader(IMAGE_FOLDER, INPUT_DIM[:2])
# ## architecture
vae = load_model(VariationalAutoencoder, RUN_FOLDER)
#vae.encoder.summary()
#vae.decoder.summary()
keypress_next_images(df=files_by_attribute, imageLoader=imageLoader, n_to_show=4, scale_value=SCALE_VALUE)

# ######
# ## reconstructing faces
# ######
np.array(files_by_attribute)[0,0]
data_flow_generic = imageLoader.build(df=files_by_attribute, 
                                      number_to_show=10,
                                      label=None, 
                                      scale_value = SCALE_VALUE)
example_batch = next(data_flow_generic)
example_images = example_batch[0]
example_images = example_batch[0]
print("Get z vectors from some random training images...")
_,_,z_points = vae.encoder.predict(example_images)
print("Recode those images from the z-vectors...")
reconst_images = vae.decoder.predict(z_points)
n_to_show=10
compare_images(n_to_show, example_images, reconst_images)

# ######
# ## Latent space distribution
# ######
print("Make 20 batches of 10 = 200 predictions:")
_,_,z_test = vae.encoder.predict(data_flow_generic, steps = 20, verbose = 1) # 20 batches * 10 samples = 200 predictions of 200-length vectors (200x200)
x = np.linspace(-3, 3, 100)
fig = plt.figure(figsize=(20, 20))
fig.subplots_adjust(hspace=0.6, wspace=0.4)
for i in range(50):
    ax = fig.add_subplot(5, 10, i+1)
    ax.hist(z_test[:,i], density=True, bins = 20)
    ax.axis('off')
    ax.text(0.5, -0.35, str(i), fontsize=10, ha='center', transform=ax.transAxes)
    ax.plot(x,norm.pdf(x))
plt.suptitle("Distribution of the latent space dimensions should be normal; showing first 50 dimensions of 200")
plt.show()
vector_distribution(z_test)

# ######
# ### Newly generated faces
# ######
n_to_show = 30
znew = np.random.normal(size = (n_to_show,vae.z_dim))
reconst = vae.decoder.predict(np.array(znew))
fig = plt.figure(figsize=(18, 5))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(n_to_show):
    ax = fig.add_subplot(3, 10, i+1)
    ax.imshow(reconst[i, :,:,:])
    ax.axis('off')
plt.suptitle("Novel, generated faces")
plt.show()

# ######
# ### Faces with an added feature
# ######
BATCH_SIZE = 500
#for label in ['Smiling','Attractive','Mouth_slightly_Open','Wearing_Lipstick','High_Cheekbones','Male','Eyeglasses','Blond_Hair']:
for label in ['Smiling']:
    feature_vec = label2z(label, BATCH_SIZE, imageLoader, files_by_attribute, vae)
    example_batch = next(data_flow_generic)
    add_vector_to_images(feature_vec, example_batch, vae, label=label)

# ######
# # Morphed faces
# ######
for start_image_file, end_image_file in [('000238.jpg', '000193.jpg'), # glasses
                                         ('000112.jpg', '000258.jpg'),
                                         ('000230.jpg', '000712.jpg')]:
    morph_faces(df=files_by_attribute, 
                imageLoader=imageLoader,
                vae=vae, 
                start_image_file=start_image_file, 
                end_image_file=end_image_file)

import os
import random
import pandas as pd
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import load_data

IMG_H = 128
IMG_W = 128


def color_randomization(image):
    # Function adds random noise to channels of colorful image
    for ch in range(image.shape[2]):
        sign = 1 if random.random() < 0.5 else -1
        noise = np.ones(image.shape[:2]) * sign * random.random() * 0.1
        image[:, :, ch] = image[:, :, ch] + noise
    return image


seed = 512
random.seed = seed

cwd = os.getcwd()

# Paths to dirs for augmented data
TRAIN_AUG_IMG_DIR = os.path.join(cwd, 'data', 'train', 'augmented_images')
TRAIN_AUG_MSK_DIR = os.path.join(cwd, 'data', 'train', 'augmented_masks')

# Creation of dirs for augmented data
try:
    os.mkdir(TRAIN_AUG_IMG_DIR)
    os.mkdir(TRAIN_AUG_MSK_DIR)
except FileExistsError:
    pass

# Paths to dirs with train data
TRAIN_IMG_DIR = os.path.join(cwd, 'data', 'train', 'images')
TRAIN_MSK_DIR = os.path.join(cwd, 'data', 'train', 'masks')

# Clusters names obtained through EDA
clusters = ['A', 'B', 'C', 'D']

# Creation of dict with names of images by cluster
names_by_cluster = {}
for c in clusters:
    names_by_cluster[c] = list(pd.read_csv(os.path.join(cwd, 'data', 'train', f'cluster_{c}.csv'),
                                           header=None)[0])

# Number of images that will belong to each cluster after augmentation
extended_size = 500

# Number of elements to add to each cluster for extended size
addition_to_cluster = [extended_size - len(names_by_cluster[c]) for c in clusters]
addition_to_cluster = [val if val > 0 else 0 for val in addition_to_cluster]

# Loop that augments and save images for each cluster
for cluster, n_aug in zip(('A', 'B', 'C', 'D'), addition_to_cluster):
    names_for_aug = names_by_cluster[cluster]
    n_img_aug = len(names_for_aug)

    X_train, Y_train = load_data(TRAIN_IMG_DIR, TRAIN_MSK_DIR, names=names_by_cluster[cluster])
    Y_train = Y_train.reshape(n_img_aug, IMG_H, IMG_W, 1)

    data_gen_args = dict(rotation_range=90,
                         width_shift_range=0.3,
                         height_shift_range=0.3,
                         zoom_range=0.3,
                         shear_range=0.3,
                         fill_mode='reflect',
                         horizontal_flip=True,
                         vertical_flip=True)

    X_datagen = ImageDataGenerator(preprocessing_function=color_randomization, **data_gen_args)
    Y_datagen = ImageDataGenerator(**data_gen_args)

    X_datagen.fit(X_train, augment=True, seed=seed)
    Y_datagen.fit(Y_train, augment=True, seed=seed)

    X_augmented = X_datagen.flow(X_train, batch_size=1, seed=seed, save_to_dir=TRAIN_AUG_IMG_DIR)
    Y_augmented = Y_datagen.flow(Y_train, batch_size=1, seed=seed, save_to_dir=TRAIN_AUG_MSK_DIR)

    for i in range(n_aug):
        X_augmented.next()
        Y_augmented.next()

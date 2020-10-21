import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.keras.losses import BinaryCrossentropy
from model import get_unet
from dice_score import dice_coef, dice_loss
from utils import load_data

seed = 11
random.seed = seed
np.random.seed(seed=seed)
tf.random.set_seed(seed)

cwd = os.getcwd()

# Create dir for model saving
try:
    os.mkdir(os.path.join(cwd, 'model'))
except FileExistsError:
    pass

# Paths to dirs with train data
train_img_dir = os.path.join(cwd, 'data', 'train', 'images')
train_msk_dir = os.path.join(cwd, 'data', 'train', 'masks')

# Paths to dirs with augmented train data
train_aug_img_dir = os.path.join(cwd, 'data', 'train', 'augmented_images')
train_aug_msk_dir = os.path.join(cwd, 'data', 'train', 'augmented_masks')

# Finding out number of train and augmented train datasets
n_train = len(os.listdir(train_img_dir))
n_train_aug = len(os.listdir(train_aug_img_dir))

# Loading train data
X_train, Y_train = load_data(train_img_dir, train_msk_dir)

# Loading augmented train data
X_train_aug, Y_train_aug = load_data(train_aug_img_dir, train_aug_msk_dir)

# Concatenation train and augmented train data to single dataset
X_train_full = np.concatenate((X_train, X_train_aug), axis=0),
Y_train_full = np.concatenate((Y_train, Y_train_aug), axis=0),

# Creation of callbacks
callbacks = [ModelCheckpoint(os.path.join(cwd, 'model', 'unet.h5'), save_best_only=1),
             EarlyStopping(patience=3, verbose=1),
             CSVLogger(os.path.join(cwd, 'model', 'log.csv'))]

# U-Net creation
unet = get_unet(n_filters=16, batchnorm=False)
unet.compile(optimizer='adam',
             loss=dice_loss,
             metrics=[dice_coef])
unet.summary()

# Training model
history = unet.fit(X_train_full, Y_train_full,
                   validation_split=0.1,
                   epochs=10, batch_size=16, shuffle=True,
                   callbacks=callbacks,
                   verbose=1)

# Plotting training history
plt.plot(history.history['dice_coef'])
plt.plot(history.history['val_dice_coef'])
plt.title('model dice score')
plt.ylabel('dice_score')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

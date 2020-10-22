import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler
from model import get_unet
from dice_score import dice_coef, dice_loss
from utils import load_data

seed = 15
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
TRAIN_IMG_DIR = os.path.join(cwd, 'data', 'train', 'images')
TRAIN_MSK_DIR = os.path.join(cwd, 'data', 'train', 'masks')

# Paths to dirs with augmented train data
TRAIN_AUG_IMG_DIR = os.path.join(cwd, 'data', 'train', 'augmented_images')
TRAIN_AUG_MSK_DIR = os.path.join(cwd, 'data', 'train', 'augmented_masks')

# Finding out number of train and augmented train datasets
n_train = len(os.listdir(TRAIN_IMG_DIR))
n_train_aug = len(os.listdir(TRAIN_AUG_IMG_DIR))

# Loading train data
X_train, Y_train = load_data(TRAIN_IMG_DIR, TRAIN_MSK_DIR)

# Loading augmented train data
X_train_aug, Y_train_aug = load_data(TRAIN_AUG_IMG_DIR, TRAIN_AUG_MSK_DIR)

# Concatenation train and augmented train data to single dataset
X_train_full = np.concatenate((X_train, X_train_aug), axis=0)
Y_train_full = np.concatenate((Y_train, Y_train_aug), axis=0)


# Creation of callbacks
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    elif epoch < 20:
        return 1e-4
    else:
        return 1e-5


callbacks = [ModelCheckpoint(os.path.join(cwd, 'model', 'unet.h5'), save_best_only=1),
             LearningRateScheduler(scheduler),
             CSVLogger(os.path.join(cwd, 'model', 'log.csv'))]

# U-Net creation
unet = get_unet(n_filters=16, batchnorm=False)
unet.compile(optimizer=Adam(learning_rate=1e-3),
             loss=dice_loss,
             metrics=[dice_coef])
unet.summary()

# Training model
history = unet.fit(X_train_full, Y_train_full,
                   validation_split=0.1,
                   epochs=25,
                   batch_size=16,
                   shuffle=True,
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

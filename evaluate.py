import os

from tensorflow.keras.models import load_model
from dice_score import dice_coef, dice_loss
from utils import load_data, save_data

IMG_H = 128
IMG_W = 128

cwd = os.getcwd()

# Paths to dirs with train data
TRAIN_IMG_DIR = os.path.join(cwd, 'data', 'train', 'images')
TRAIN_MSK_DIR = os.path.join(cwd, 'data', 'train', 'masks')
TRAIN_PRED_DIR = os.path.join(cwd, 'data', 'train', 'predicted')

# Paths to dirs with validation (internal test) data
VALID_IMG_DIR = os.path.join(cwd, 'data', 'validation', 'images')
VALID_MSK_DIR = os.path.join(cwd, 'data', 'validation', 'masks')
VALID_PRED_DIR = os.path.join(cwd, 'data', 'validation', 'predicted')

# Paths to dir with test data
TEST_IMG_DIR = os.path.join(cwd, 'data', 'test', 'images')
TEST_PRED_DIR = os.path.join(cwd, 'data', 'test', 'predicted')

# Lists of names of images per datasets
train_img_names = os.listdir(TRAIN_IMG_DIR)
valid_img_names = os.listdir(VALID_IMG_DIR)
test_img_names = os.listdir(TEST_IMG_DIR)

# Number of images per datasets
n_train = len(train_img_names)
n_valid = len(valid_img_names)
n_test = len(test_img_names)

# Loading training and validation data
X_train, Y_train = load_data(TRAIN_IMG_DIR, TRAIN_MSK_DIR)
X_valid, Y_valid = load_data(VALID_IMG_DIR, VALID_MSK_DIR)
X_test = load_data(TEST_IMG_DIR)

# Loading trained model
unet = load_model(os.path.join(cwd, 'model', 'unet.h5'), custom_objects={'dice_loss': dice_loss,
                                                                         'dice_coef': dice_coef})

# Evaluation model on train data and saving obtained masks
print('On train', unet.evaluate(X_train, Y_train))
Y_train_predicted = unet.predict(X_train).reshape(n_train, IMG_H, IMG_W)
save_data(TRAIN_PRED_DIR, Y_train_predicted, train_img_names)

# Evaluation model on validation data and saving obtained masks
print('On validation', unet.evaluate(X_valid, Y_valid))
Y_valid_predicted = unet.predict(X_valid).reshape(n_valid, IMG_H, IMG_W)
save_data(VALID_PRED_DIR, Y_valid_predicted, valid_img_names)

# Predicting masks on test data
Y_test_predicted = unet.predict(X_test).reshape(n_test, IMG_H, IMG_W)
save_data(TEST_PRED_DIR, Y_test_predicted, test_img_names)

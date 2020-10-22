import os

from tensorflow.keras.models import load_model
from dice_score import dice_coef, dice_loss
from utils import load_data, save_data

IMG_H = 128
IMG_W = 128

cwd = os.getcwd()

# Paths to dir with test data
test_img_dir = os.path.join(cwd, 'data', 'test', 'images')

# Lists of names of images per datasets
test_img_names = os.listdir(test_img_dir)

# Number of images per datasets
n_test = len(test_img_names)

# Loading training and validation data
X_test = load_data(test_img_dir)

# Loading trained model
unet = load_model(os.path.join(cwd, 'model', 'unet.h5'), custom_objects={'dice_loss': dice_loss,
                                                                         'dice_coef': dice_coef})

# Predicting masks on test data
Y_test_predicted = unet.predict(X_test).reshape(n_test, IMG_H, IMG_W)
save_data(os.path.join(cwd, 'data', 'test', 'predicted'), Y_test_predicted, test_img_names)

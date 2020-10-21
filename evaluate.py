import os

from tensorflow.keras.models import load_model
from dice_score import dice_coef, dice_loss
from utils import load_data, save_data


cwd = os.getcwd()

# Paths to dirs with train data
train_img_dir = os.path.join(cwd, 'data', 'train', 'images')
train_msk_dir = os.path.join(cwd, 'data', 'train', 'masks')

# Paths to dirs with validation (internal test) data
valid_img_dir = os.path.join(cwd, 'data', 'validation', 'images')
valid_msk_dir = os.path.join(cwd, 'data', 'validation', 'masks')

# Paths to dir with test data
test_img_dir = os.path.join(cwd, 'data', 'test', 'images')

# Lists of names of images per datasets
train_img_names = os.listdir(train_img_dir)
valid_img_names = os.listdir(valid_img_dir)
test_img_names = os.listdir(test_img_dir)

# Number of images per datasets
n_train = len(train_img_names)
n_valid = len(valid_img_names)
n_test = len(test_img_names)

# Loading training and validation data
X_train, Y_train = load_data(train_img_dir, train_msk_dir)
X_valid, Y_valid = load_data(valid_img_dir, valid_msk_dir)
X_test = load_data(test_img_dir)

# Loading trained model
unet = load_model(os.path.join(cwd, 'model', 'unet.h5'), custom_objects={'dice_loss': dice_loss,
                                                                         'dice_coef': dice_coef})

# Evaluation model on train data and saving obtained masks
print('On train', unet.evaluate(X_train, Y_train))
Y_train_predicted = unet.predict(X_train).reshape(n_train, 128, 128)
save_data(os.path.join(cwd, 'data', 'train', 'predicted'), Y_train_predicted, train_img_names)

# Evaluation model on validation data and saving obtained masks
print('On validation', unet.evaluate(X_valid, Y_valid))
Y_valid_predicted = unet.predict(X_valid).reshape(n_valid, 128, 128)
save_data(os.path.join(cwd, 'data', 'validation', 'predicted'), Y_valid_predicted, valid_img_names)

# Predicting masks on test data
Y_test_predicted = unet.predict(X_test).reshape(n_test, 128, 128)
save_data(os.path.join(cwd, 'data', 'test', 'predicted'), Y_test_predicted, test_img_names)

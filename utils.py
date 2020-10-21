import os
import sys
import cv2
import numpy as np

from tqdm import tqdm


def load_data(img_dir_path, msk_dir_path=None, names=None):
    sys.stdout.flush()
    if names is not None:
        list_of_names = names
    else:
        list_of_names = os.listdir(img_dir_path)

    n = len(list_of_names)

    X_train = np.zeros((n, 128, 128, 3))
    if msk_dir_path is not None:
        Y_train = np.zeros((n, 128, 128))

    for i, img_name in tqdm(enumerate(list_of_names), total=n):
        img = cv2.imread(os.path.join(img_dir_path, img_name))
        img_rs = cv2.resize(img, (128, 128))
        X_train[i] = img_rs / 255
        if msk_dir_path is not None:
            img = cv2.imread(os.path.join(msk_dir_path, img_name), 0)
            Y_train[i] = img / 255
    if msk_dir_path is not None:
        return X_train, Y_train
    return X_train


def save_data(destination_path, Y_predicted, names):
    n = len(names)
    for i, img_name in tqdm(enumerate(names), total=n):
        img = ((Y_predicted[i] > 0.5) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(destination_path, f'{img_name}'), img)
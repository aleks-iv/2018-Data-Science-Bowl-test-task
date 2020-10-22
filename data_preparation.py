import os
import cv2
import numpy as np

from tqdm import tqdm
from sklearn.model_selection import train_test_split

IMG_H = 128
IMG_W = 128

cwd = os.getcwd()

ORIGINAL_DATA = os.path.join(cwd, 'stage1_train')
ORIGINAL_TEST_DATA = os.path.join(cwd, 'stage1_test')

DATA = os.path.join(cwd, 'data')

TRAIN_DATA = os.path.join(DATA, 'train')
TRAIN_IMAGES = os.path.join(TRAIN_DATA, 'images')
TRAIN_MASKS = os.path.join(TRAIN_DATA, 'masks')
TRAIN_PREDICT = os.path.join(TRAIN_DATA, 'predicted')

VALID_DATA = os.path.join(DATA, 'validation')
VALID_IMAGES = os.path.join(VALID_DATA, 'images')
VALID_MASKS = os.path.join(VALID_DATA, 'masks')
VALID_PREDICT = os.path.join(VALID_DATA, 'predicted')

TEST_DATA = os.path.join(DATA, 'test')
TEST_IMAGES = os.path.join(TEST_DATA, 'images')
TEST_PREDICT = os.path.join(VALID_DATA, 'predicted')

directories = [DATA,
               TRAIN_DATA,
               TRAIN_IMAGES,
               TRAIN_MASKS,
               TRAIN_PREDICT,
               VALID_DATA,
               VALID_IMAGES,
               VALID_MASKS,
               VALID_PREDICT,
               TEST_DATA,
               TEST_IMAGES,
               TEST_PREDICT]

for path in directories:
    try:
        os.mkdir(path)
    except FileExistsError:
        pass


def create_single_mask(img_dir_name, origin_path):
    new_mask = np.zeros((IMG_H, IMG_W, 1))
    original_masks_path = os.path.join(origin_path, img_dir_name, 'masks')
    original_masks_list = os.listdir(original_masks_path)
    for mask_name in original_masks_list:
        mask_path = os.path.join(original_masks_path, mask_name)
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (IMG_H, IMG_W))
        mask = np.expand_dims(mask, axis=-1)
        new_mask = np.maximum(new_mask, mask)
    return new_mask


def img_placement(img_dir_names, origin_path, destination_path):
    n = len(img_dir_names)
    for img_dir_name in tqdm(img_dir_names, total=n):
        image_name = img_dir_name + '.png'
        img = cv2.imread(os.path.join(origin_path, img_dir_name, 'images', image_name))
        img_path_name = os.path.join(destination_path, image_name)
        cv2.imwrite(img_path_name, img)


def single_masks_placement(img_dir_names, origin_path, destination_path):
    n = len(img_dir_names)

    for img_dir_name in tqdm(img_dir_names, total=n):
        single_mask = create_single_mask(img_dir_name, origin_path)
        mask_name = img_dir_name + '.png'
        single_mask_path_name = os.path.join(destination_path, mask_name)
        cv2.imwrite(single_mask_path_name, single_mask.astype(np.uint8))


img_names = os.listdir(ORIGINAL_DATA)
img_test_names = os.listdir(ORIGINAL_TEST_DATA)

img_train_names, img_valid_names = train_test_split(img_names, test_size=0.1)

print('Train data single mask creation and relocation')
img_placement(img_train_names, ORIGINAL_DATA, TRAIN_IMAGES)
single_masks_placement(img_train_names, ORIGINAL_DATA, TRAIN_MASKS)

print('Validation data single mask creation and relocation')
img_placement(img_valid_names, ORIGINAL_DATA, VALID_IMAGES)
single_masks_placement(img_valid_names, ORIGINAL_DATA, VALID_MASKS)

print('Test data relocation')
img_placement(img_test_names, ORIGINAL_TEST_DATA, TEST_IMAGES)

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
TRAIN_DATA_IMAGES = os.path.join(TRAIN_DATA, 'images')
TRAIN_DATA_MASKS = os.path.join(TRAIN_DATA, 'masks')

VALID_DATA = os.path.join(DATA, 'validation')
VALID_DATA_IMAGES = os.path.join(VALID_DATA, 'images')
VALID_DATA_MASKS = os.path.join(VALID_DATA, 'masks')

TEST_DATA = os.path.join(DATA, 'test')
TEST_DATA_IMAGES = os.path.join(TEST_DATA, 'images')

directories = [DATA,
               TRAIN_DATA,
               TRAIN_DATA_IMAGES,
               TRAIN_DATA_MASKS,
               VALID_DATA,
               VALID_DATA_IMAGES,
               VALID_DATA_MASKS,
               TEST_DATA,
               TEST_DATA_IMAGES]

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
img_placement(img_train_names, ORIGINAL_DATA, TRAIN_DATA_IMAGES)
single_masks_placement(img_train_names, ORIGINAL_DATA, TRAIN_DATA_MASKS)

print('Validation data single mask creation and relocation')
img_placement(img_valid_names, ORIGINAL_DATA, VALID_DATA_IMAGES)
single_masks_placement(img_valid_names, ORIGINAL_DATA, VALID_DATA_MASKS)

print('Test data relocation')
img_placement(img_test_names, ORIGINAL_TEST_DATA, TEST_DATA_IMAGES)

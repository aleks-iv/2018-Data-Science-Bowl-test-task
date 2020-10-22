from tensorflow.keras import backend as K


def dice_coef(y_true, y_pred):
    # Dice metric
    intersection = K.sum(y_true * y_pred, axis=(1, 2))
    union = K.sum(y_true, axis=(1, 2)) + K.sum(y_pred, axis=(1, 2))
    dice = (2.0 * intersection + 1.0) / (union + 1.0)
    return K.mean(dice, axis=0)


def dice_coef_single(y_true, y_pred):
    # Dice metric for single example evaluation
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred)
    dice = (2. * intersection) / union
    return dice


def dice_loss(y_true, y_pred):
    # Dice loss function
    return 1 - dice_coef(y_true, y_pred)

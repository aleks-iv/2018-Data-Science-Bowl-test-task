from tensorflow import keras
from tensorflow.keras import layers


def conv2d_block(input_tensor, n_filters, kernel_size=3, kernel_initializer='he_normal', batchnorm=True):
    # first layer
    x = layers.Conv2D(filters=n_filters, kernel_size=kernel_size, kernel_initializer=kernel_initializer,
                      padding='same')(input_tensor)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    # second layer
    x = layers.Conv2D(filters=n_filters, kernel_size=kernel_size, kernel_initializer=kernel_initializer,
                      padding='same')(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x


def get_unet(n_filters=16, kernel_initializer='he_normal', batchnorm=True):
    inputs = layers.Input((128, 128, 3))

    # contracting path
    c1 = conv2d_block(inputs, n_filters=n_filters, kernel_initializer=kernel_initializer, batchnorm=batchnorm)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_initializer=kernel_initializer, batchnorm=batchnorm)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_initializer=kernel_initializer, batchnorm=batchnorm)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_initializer=kernel_initializer, batchnorm=batchnorm)
    p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_initializer=kernel_initializer, batchnorm=batchnorm)

    # expansive path
    u6 = layers.Conv2DTranspose(n_filters * 8, kernel_size=3, strides=2, padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_initializer=kernel_initializer, batchnorm=batchnorm)

    u7 = layers.Conv2DTranspose(n_filters * 4, kernel_size=3, strides=2, padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_initializer=kernel_initializer, batchnorm=batchnorm)

    u8 = layers.Conv2DTranspose(n_filters * 2, kernel_size=3, strides=2, padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_initializer=kernel_initializer, batchnorm=batchnorm)

    u9 = layers.Conv2DTranspose(n_filters * 1, kernel_size=3, strides=2, padding='same')(c8)
    u9 = layers.concatenate([u9, c1], axis=3)
    c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_initializer=kernel_initializer, batchnorm=batchnorm)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    unet = keras.Model(inputs=[inputs], outputs=[outputs])
    return unet

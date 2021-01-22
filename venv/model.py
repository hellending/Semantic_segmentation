import tensorflow as tf
from tensorflow.keras.models import save_model, load_model, Model
from tensorflow.keras.layers import Input, Dropout, BatchNormalization, LeakyReLU, concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Conv2DTranspose
#U-net 模型
def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same'):
    x = Conv2D(nb_filter, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x


def Conv2dT_BN(x, filters, kernel_size, strides=(2, 2), padding='same'):
    x = Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x
def unet():
    img_w = 128
    img_h = 128
    #实例化一个张量
    inputs = Input((img_w, img_h, 3))
#
#     conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
#     conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(conv1)
#     pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
#
#     conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(pool1)
#     conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(conv2)
#     pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
#
#     conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(pool2)
#     conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(conv3)
#     pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
#
#     conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(pool3)
#     conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(conv4)
#     pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
#
#     conv5 = tf.keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same")(pool4)
#     conv5 = tf.keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same")(conv5)
#
#     up6 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
#     conv6 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(up6)
#     conv6 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(conv6)
#
#     up7 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
#     conv7 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(up7)
#     conv7 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(conv7)
#
#     up8 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
#     conv8 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(up8)
#     conv8 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(conv8)
#
#     up9 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
#     conv9 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(up9)
#     conv9 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(conv9)
#
#     conv10 = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid")(conv9)
    #conv10 = Conv2D(n_label, (1, 1), activation="softmax")(conv9)

    #
    conv1 = Conv2d_BN(inputs, 32, (3, 3))
    conv1 = Conv2d_BN(conv1, 32, (3, 3))
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)

    conv2 = Conv2d_BN(pool1, 64, (3, 3))
    conv2 = Conv2d_BN(conv2, 64, (3, 3))
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)

    conv3 = Conv2d_BN(pool2, 128, (3, 3))
    conv3 = Conv2d_BN(conv3, 128, (3, 3))
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)

    conv4 = Conv2d_BN(pool3, 256, (3, 3))
    conv4 = Conv2d_BN(conv4, 256, (3, 3))
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv4)

    conv5 = Conv2d_BN(pool4, 256, (3, 3))
    # conv5 = Dropout(0.5)(conv5)
    conv5 = Conv2d_BN(conv5, 256, (3, 3))
    # conv5 = Dropout(0.5)(conv5)

    convt1 = Conv2dT_BN(conv5, 128, (3, 3))
    concat1 = concatenate([conv4, convt1], axis=3)
    # concat1 = Dropout(0.5)(concat1)
    conv6 = Conv2d_BN(concat1, 128, (3, 3))
    conv6 = Conv2d_BN(conv6, 128, (3, 3))

    convt2 = Conv2dT_BN(conv6, 64, (3, 3))
    concat2 = concatenate([conv3, convt2], axis=3)
    # concat2 = Dropout(0.5)(concat2)
    conv7 = Conv2d_BN(concat2, 64, (3, 3))
    conv7 = Conv2d_BN(conv7, 64, (3, 3))
    convt3 = Conv2dT_BN(conv7, 32, (3, 3))
    concat3 = concatenate([conv2, convt3], axis=3)
    # concat3 = Dropout(0.5)(concat3)
    conv8 = Conv2d_BN(concat3, 32, (3, 3))
    conv8 = Conv2d_BN(conv8, 32, (3, 3))

    convt4 = Conv2dT_BN(conv8, 16, (3, 3))
    concat4 = concatenate([conv1, convt4], axis=3)
    # concat4 = Dropout(0.5)(concat4)
    conv9 = Conv2d_BN(concat4, 16, (3, 3))
    conv9 = Conv2d_BN(conv9, 16, (3, 3))
    # conv9 = Dropout(0.5)(conv9)
    outpt = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=outpt)
    model.compile(optimizer='Nadam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

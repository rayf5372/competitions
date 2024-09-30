import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dummy directories containing the training and validation data.
train_dir = '/Users/ft4/Desktop/data/train'
validation_dir = '/Users/ft4/Desktop/data/validation'

batch_size = 32
img_height = 150
img_weight = 150

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_dir,
    batch_size=batch_size,
    image_size=(img_weight, img_height),
    seed=123,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(img_weight, img_height),
    batch_size=batch_size,
)

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(150, 150), batch_size=20, class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir, target_size=(150, 150), batch_size=32, class_mode='binary'
)


# Using buffered prefetching.
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Standarizing the data.

normalization_layer = layers.experimental.preprocessing.Rescaling(1.0 / 255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]

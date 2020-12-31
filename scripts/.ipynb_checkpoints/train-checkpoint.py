import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
                                    Dense,
                                    Conv2D,
                                    MaxPool2D,
                                    BatchNormalization,
                                    Flatten,
                                    Dropout
                                    )
train_dir = "dataset/train"
test_dir = "dataset/test"

BATCH_SIZE = 64
IMG_SIZE = 224
EPOCHS = 15
lr = 0.00003
(b1, b2) = (0.999, 0.99)
CLASSES = ["mask", "no mask"]


train_datagen = ImageDataGenerator(
    rotation_range=20,
    horizontal_flip=True,
    featurewise_center=True, 
    featurewise_std_normalization=True,
)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(IMG_SIZE, IMG_SIZE), class_mode='binary', batch_size=BATCH_SIZE)


test_datagen = ImageDataGenerator(
    rotation_range=20,
    horizontal_flip=True,
    featurewise_center=True, 
    featurewise_std_normalization=True,
)
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(IMG_SIZE, IMG_SIZE), class_mode='binary', batch_size=BATCH_SIZE)

vgg = tf.keras.applications.VGG16(include_top=False, input_shape=(224,224,3), pooling='avg')

for layer in vgg.layers:
    layer.trainable = False

model = Sequential([
    vgg,
    Dense(1, activation='sigmoid')
])
model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['acc'])
model.fit_generator(train_generator, steps_per_epoch=len(train_generator), workers=4, epochs=EPOCHS)
model.evaluate_generator(test_generator)
model.save("mask.h5")
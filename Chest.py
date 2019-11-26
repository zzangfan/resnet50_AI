import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from keras.preprocessing.image import ImageDataGenerator
import cv2
from Resnet50_re import Resnet_re

DIR ="D:/chest-xray-pneumonia/chest_xray/chest_xray/"
TRAIN_DIR = DIR+"train/"
TEST_DIR = DIR+"test/"
VAL_DIR = DIR+"val/"

batch_size =300
epochs=20
IMG_HEIGH = 224
IMG_WIDHT =224


train_image_gen = ImageDataGenerator(rescale=1./255,
                                     )

val_image_en = ImageDataGenerator(rescale=1./255,
                                     )


test_image_gen = ImageDataGenerator(rescale=1./255,
                                     )

train_gen = train_image_gen.flow_from_directory(directory=TRAIN_DIR,
                                                class_mode='binary',
                                                color_mode='grayscale',
                                                shuffle=True,
                                                target_size=(IMG_HEIGH, IMG_WIDHT))

valid_gen = val_image_en.flow_from_directory(directory=VAL_DIR,
                                             class_mode='binary',
                                             color_mode='grayscale',
                                             shuffle=True,
                                             target_size=(IMG_HEIGH, IMG_WIDHT))

test_gen = test_image_gen.flow_from_directory(directory=TEST_DIR,
                                              class_mode='binary',
                                              color_mode='grayscale',
                                              shuffle=True,

                                              target_size=(IMG_HEIGH, IMG_WIDHT))


r50 = Resnet_re(classes=1,dim=(224,224,1))
output_tensor = r50.main()
model = r50.model(r50.input_tensor,output_tensor)

model.compile(optimizer='adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

history = model.fit_generator(train_gen,
                              steps_per_epoch=10,
                              epochs=epochs,
                              validation_data=valid_gen,

                              validation_steps=10)


model.save('ChestResnet50.h5')
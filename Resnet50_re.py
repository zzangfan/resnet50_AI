from keras import models , layers
from keras import Input
from keras.models import Model, load_model
from keras.preprocessing.image import  ImageDataGenerator
from keras import optimizers, initializers, regularizers, metrics
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import BatchNormalization, Conv2D, Activation, Dense, \
    GlobalAveragePooling2D, MaxPool2D, ZeroPadding2D, Add

import os
import matplotlib.pyplot as plt
import numpy as np
import math



class Resnet_re:

    def __init__(self,classes = 6,dim=(224,224,3)):
        self.classes = classes
        self.dim = dim
        self.input_tensor = Input(shape=self.dim, dtype='float32', name='input')



    def main(self):



        x = ZeroPadding2D(padding=(3,3))(self.input_tensor)
        x = Conv2D(64,(7,7), strides =(2,2))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = ZeroPadding2D(padding=(1,1))(x)
        #5 layers
        x = MaxPool2D((3,3),2)(x)
        shortcut = x
        #7 layers

        x = Conv2D(64,(1,1),strides=(1,1),padding='valid')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(64,(3,3),strides=(1,1),padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(256,(1,1),strides=(1,1),padding='valid')(x)
        shortcut = Conv2D(256,(1,1),strides=(1,1),padding='valid')(shortcut)
        x = BatchNormalization()(x)
        shortcut = BatchNormalization()(shortcut)
        x = Add()([x,shortcut])
        x = Activation('relu')(x)
        #18 layers

        shortcut = x

        x = Conv2D(64,(1,1), strides=(1,1),padding='valid')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(64,(3,3), strides=(1,1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(256,(1,1),strides=(1,1),padding='valid')(x)
        x = BatchNormalization()(x)
        x = Add()([x,shortcut])
        x = Activation('relu')(x)

        shortcut = x
        #28 layers
        x = Conv2D(64,(1,1),strides=(1,1),padding='valid')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(64,(3,3),strides=(1,1),padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(256,(1,1),strides=(1,1),padding='valid')(x)
        x = BatchNormalization()(x)
        x = Add()([x,shortcut])
        x = Activation('relu')(x)

        shortcut = x
        #38 layers

        x = Conv2D(128,(1,1),strides=(2,2),padding='valid')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(128,(3,3),strides=(1,1),padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(512,(1,1),strides=(1,1),padding='valid')(x)
        shortcut = Conv2D(512,(1,1),strides=(2,2),padding='valid')(shortcut)
        x = BatchNormalization()(x)
        shortcut = BatchNormalization()(shortcut)
        x = Add()([x,shortcut])
        x = Activation('relu')(x)

        shortcut = x

        #50 layers


        x = Conv2D(128,(1,1),strides=(1,1),padding='valid')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(128,(3,3),strides=(1,1),padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(512,(1,1),strides=(1,1),padding='valid')(x)
        x = BatchNormalization()(x)
        x = Add()([x,shortcut])
        x = Activation('relu')(x)

        shortcut = x

        #60 layers


        x = Conv2D(128,(1,1),strides=(1,1),padding='valid')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(128,(3,3),strides=(1,1),padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(512,(1,1),strides=(1,1),padding='valid')(x)
        x = BatchNormalization()(x)
        x = Add()([x,shortcut])
        x = Activation('relu')(x)

        shortcut = x
        #70 layers

        x = Conv2D(128,(1,1),strides=(1,1),padding='valid')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(128,(3,3),strides=(1,1),padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(512,(1,1),strides=(1,1),padding='valid')(x)
        x = BatchNormalization()(x)
        x = Add()([x,shortcut])
        x = Activation('relu')(x)

        shortcut = x

        #80 layers


        x = Conv2D(256,(1,1),strides=(2,2),padding='valid')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(256,(3,3),strides=(1,1),padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(1024,(1,1),strides=(1,1),padding='valid')(x)
        shortcut = Conv2D(1024,(1,1),strides=(2,2),padding='valid')(shortcut)
        shortcut = BatchNormalization()(shortcut)
        x = BatchNormalization()(x)
        x = Add()([x,shortcut])
        x = Activation('relu')(x)

        shortcut = x
        #102 layers


        x = Conv2D(256,(1,1),strides=(1,1),padding='valid')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(256,(3,3),strides=(1,1),padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(1024,(1,1),strides=(1,1),padding='valid')(x)
        x = BatchNormalization()(x)
        x = Add()([x,shortcut])
        x = Activation('relu')(x)

        shortcut = x

        #112 layers


        x = Conv2D(256,(1,1),strides=(1,1),padding='valid')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(256,(3,3),strides=(1,1),padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(1024,(1,1),strides=(1,1),padding='valid')(x)
        x = BatchNormalization()(x)
        x = Add()([x,shortcut])
        x = Activation('relu')(x)

        shortcut = x

        #122 layers

        x = Conv2D(256,(1,1),strides=(1,1),padding='valid')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(256,(3,3),strides=(1,1),padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(1024,(1,1),strides=(1,1),padding='valid')(x)
        x = BatchNormalization()(x)
        x = Add()([x,shortcut])
        x = Activation('relu')(x)

        shortcut = x

        #132 layers

        x = Conv2D(256,(1,1),strides=(1,1),padding='valid')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(256,(3,3),strides=(1,1),padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(1024,(1,1),strides=(1,1),padding='valid')(x)
        x = BatchNormalization()(x)
        x = Add()([x,shortcut])
        x = Activation('relu')(x)

        shortcut = x

        #142 layers

        x = Conv2D(512,(1,1),strides=(2,2),padding='valid')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(512,(3,3),strides=(1,1),padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(2048,(1,1),strides=(1,1),padding='valid')(x)
        shortcut = Conv2D(2048,(1,1),strides=(2,2),padding='valid')(shortcut)
        shortcut = BatchNormalization()(shortcut)
        x = BatchNormalization()(x)
        x = Add()([x,shortcut])
        x = Activation('relu')(x)

        shortcut = x

        #154 layers

        x = Conv2D(512,(1,1),strides=(1,1),padding='valid')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(512,(3,3),strides=(1,1),padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(2048,(1,1),strides=(1,1),padding='valid')(x)
        x = BatchNormalization()(x)
        x = Add()([x,shortcut])
        x = Activation('relu')(x)

        shortcut = x

        #164 layers

        x = Conv2D(512,(1,1),strides=(1,1),padding='valid')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(512,(3,3),strides=(1,1),padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid')(x)
        x = BatchNormalization()(x)
        x = Add()([x,shortcut])
        x = Activation('relu')(x)

        x = GlobalAveragePooling2D()(x)
        output_tensor = Dense(self.classes,activation='softmax')(x)
        print('output_tensor')
        return output_tensor

    def model(self,input_tensor, output_tensor):
        resnet50 = Model(input_tensor,output_tensor)
        return resnet50










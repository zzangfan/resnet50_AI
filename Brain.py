from Resnet50_re import Resnet_re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import pydicom
import os
import cv2
from sklearn.utils import shuffle
from tensorflow.python.keras.utils.data_utils import Sequence




DIR = "D:/rsna-intracranial-hemorrhage-detection/"
TRAIN_DIR = DIR+"stage_1_train_images/"
TEST_DIR = DIR+"stage_1_test_images/"

IMG_HEIGHT = 224
IMG_WIDTH = 224
epochs = 100
banch_size=300


def window_totla(load_dir_filename):
    dcm = pydicom.read_file(load_dir_filename)

    def choose_data(x):
        if type(x) == pydicom.multival.MultiValue:
            return int(x[0])
        else:
            return int(x)

    def get_windowing(data):
        dicom_fields = [data[('0028', '1050')].value,  # window center
                        data[('0028', '1051')].value,  # window width
                        data[('0028', '1052')].value,  # intercept
                        data[('0028', '1053')].value]  # slope

        return [choose_data(x) for x in dicom_fields]

    def windowing(dcm, window_center, window_width):
        _, _, intercept, slope = get_windowing(dcm)
        img = dcm.pixel_array * slope + intercept
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        img[img < img_min] = img_min
        img[img > img_max] = img_max

        return img

    def bsb_window(img):
        brain_img = windowing(img, 40, 80)
        subdural_img = windowing(img, 80, 200)
        soft_img = windowing(img, 40, 380)

        brain_img = (brain_img - 0) / 80
        subdural_img = (subdural_img - (-20)) / 200
        soft_img = (soft_img - (-150)) / 380
        bsb_img = np.array([brain_img, subdural_img, soft_img]).transpose(1, 2, 0)

        return bsb_img

    return bsb_window(dcm)


class MyDataGenerator(Sequence):

    def __init__(self, data_ID_labels, DIR=TRAIN_DIR, batch_size=100, dim=(512, 512)):

        self.dim = dim
        self.batch_size = batch_size
        self.data = data_ID_labels
        self.on_epoch_end()
        self.DIR = DIR

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):

        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        data_temp = [self.data['filename'][k] for k in indexes]
        data_label_temp = [[int(self.data['any'][i]),
                            int(self.data['epidural'][i]),
                            int(self.data['intraparenchymal'][i]),
                            int(self.data['intraventricular'][i]),
                            int(self.data['subarachnoid'][i]),
                            int(self.data['subdural'][i])] for i in indexes]

        # Generate data

        X, y = self.__data_generation(data_temp, data_label_temp)

        return X, y

    def on_epoch_end(self):

        self.indexes = np.arange(len(self.data))

    def __data_generation(self, data_temp, data_label_temp):

        X = []
        y = []

        for i, ID in enumerate(data_temp):
            FILE_DIR = self.DIR + '/' + data_temp[i]
            array = np.array(self.window_total(FILE_DIR), dtype='uint8')
            resized = cv2.resize(array, self.dim)
            X.append(resized)

        X = np.array(X, dtype='uint8').reshape(-1, self.dim[0], self.dim[1], 3)

        y_train = np.asarray(data_label_temp)
        return X, y_train

    # -------------------------------------------------------------------------------------------------------------------------------

    def window_total(self, load_dir_filename):
        dcm = pydicom.read_file(load_dir_filename)

        def choose_data(x):
            if type(x) == pydicom.multival.MultiValue:
                return int(x[0])
            else:
                return int(x)

        def get_windowing(data):
            dicom_fields = [data[('0028', '1050')].value,  # window center
                            data[('0028', '1051')].value,  # window width
                            data[('0028', '1052')].value,  # intercept
                            data[('0028', '1053')].value]  # slope

            return [choose_data(x) for x in dicom_fields]

        def windowing(dcm, window_center, window_width):
            _, _, intercept, slope = get_windowing(dcm)
            img = dcm.pixel_array * slope + intercept
            img_min = window_center - window_width // 2
            img_max = window_center + window_width // 2
            img[img < img_min] = img_min
            img[img > img_max] = img_max

            return img

        def bsb_window(img):
            brain_img = windowing(img, 40, 80)
            subdural_img = windowing(img, 80, 200)
            soft_img = windowing(img, 40, 380)

            brain_img = (brain_img - 0) / 80
            subdural_img = (subdural_img - (-20)) / 200
            soft_img = (soft_img - (-150)) / 380
            bsb_img = np.array([brain_img, subdural_img, soft_img]).transpose(1, 2, 0)

            return bsb_img

        return bsb_window(dcm)

df =  pd.read_csv("캡스톤_전처리.csv",encoding='utf8',index_col=0)
pivot_df = df.drop_duplicates().pivot(index='filename',columns='subtype',values='label')
pivot_df.to_csv("피봇캡스톤.csv",encoding='utf-8',index=False)
pivot_data = shuffle(pivot_df).reset_index()

SEED = 3

test_df = pivot_df.sample(frac=0.1,random_state=SEED).copy()
train_df = pivot_df.drop(test_df.index,axis=0)

valid_df = train_df.sample(frac=0.3,random_state=SEED).copy()
train_df = train_df.drop(valid_df.index,axis=0)

train_df.reset_index(drop=False, inplace=True)
valid_df.reset_index(drop=False, inplace=True)
test_df.reset_index(drop=False, inplace=True)

train_gen = MyDataGenerator(train_df,DIR=TRAIN_DIR,batch_size=30,dim=(IMG_HEIGHT,IMG_WIDTH))
valid_gen = MyDataGenerator(valid_df,DIR=TRAIN_DIR,batch_size=30,dim=(IMG_HEIGHT, IMG_WIDTH))
test_gen =  MyDataGenerator(test_df,DIR=TRAIN_DIR, batch_size=30,dim=(IMG_HEIGHT, IMG_WIDTH))

r50 = Resnet_re(classes=6,dim=(224,224,3))
output_tensor = r50.main()
model = r50.model(r50.input_tensor,output_tensor)
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history= model.fit_generator(generator=train_gen,steps_per_epoch=100,
                    epochs=5,validation_data=valid_gen,validation_steps=30,workers=-1)


model.save('BrainAI.h5')

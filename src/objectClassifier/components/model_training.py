import os
import cv2
import numpy as np
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from pathlib import Path
from objectClassifier import logger
from objectClassifier.entity.config_entity import TrainingConfig

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def data_generator(self):
        training_path = self.config.training_data
        validation_path = self.config.validation_data

        self.data = []
        self.labels = []

        for subdir in ['DR','No_DR']:
            subdir_path = os.path.join(training_path,subdir)

            for file in os.listdir(subdir_path):
                image_path = os.path.join(subdir_path,file)
                
                image = cv2.imread(image_path)
                image = img_to_array(image)
                
                # resizing the image according to the size specification of Xception (original size - (224,224), resized to - (128,128))
                image = cv2.resize(image, (128,128))
                self.data.append(image)

                self.labels.append(subdir)
        
        for subdir in ['DR','No_DR']:
            subdir_path = os.path.join(validation_path,subdir)

            for file in os.listdir(subdir_path):
                image_path = os.path.join(subdir_path,file)
                
                image = cv2.imread(image_path)
                image = img_to_array(image)
                
                # resizing the image according to the size specification of Xception (original size - (224,224), resized to - (128,128))
                image = cv2.resize(image, (128,128))
                self.data.append(image)

                self.labels.append(subdir)
        
        # converting the data and labels to numpy array and scaling the images
        self.data = np.asarray(self.data) / 255
        self.labels = np.asarray(self.labels)

        # Encoding the labels (to numericals)
        le = LabelEncoder()
        le.fit(self.labels)
        self.labels = to_categorical(le.transform(self.labels),2)

    def train(self):
        # Splitting the data into train, test, and valid
        x_train, x_test, y_train, y_test = train_test_split(self.data, self.labels, test_size=0.2,random_state=42)

        x_train = x_train[:-round(len(x_train)/10)]
        y_train = y_train[:-round(len(y_train)/10)]
        x_val = x_train[-round(len(x_train)/10):]
        y_val = y_train[-round(len(y_train)/10):]

        logger.info(f'Training Samples: {len(x_train)}')
        logger.info(f'Testing Samples: {len(x_test)}')
        logger.info(f'Validation Samples: {len(x_val)}')

        logger.info(f'Size of training images {x_train.shape}')
        logger.info(f'Size of testing images {x_test.shape}')
        logger.info(f'Size of validation images {x_val.shape}')

        self.model.fit(
            x_train,y_train,epochs=self.config.params_epochs,validation_data=(x_val,y_val),batch_size=32
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
    
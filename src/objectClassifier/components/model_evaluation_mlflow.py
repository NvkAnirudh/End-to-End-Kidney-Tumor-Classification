import os
import cv2
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from objectClassifier.entity.config_entity import EvaluationConfig
from objectClassifier.utils.helper import read_yaml, create_directories,save_json

class Evaluation:
    def __init__(self,config: EvaluationConfig):
        self.config = config

    def test_data_generator(self):
        testing_path = self.config.testing_data

        self.data = []
        self.labels = []

        for subdir in ['DR','No_DR']:
            subdir_path = os.path.join(testing_path,subdir)

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

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self.test_data_generator()
        self.score = self.model.evaluate(self.data, self.labels, batch_size=32)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(Path("scores.json"), data=scores)

    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({"loss": self.score[0], "accuracy": self.score[1]})

            # Model registry does not work with file store
            if tracking_url_type_store != 'file':
                mlflow.keras.log_model(self.model, 'model', registered_model_name='XceptionWithAdamOpt')
            
            else:
                mlflow.keras.log_model(self.model, 'model')


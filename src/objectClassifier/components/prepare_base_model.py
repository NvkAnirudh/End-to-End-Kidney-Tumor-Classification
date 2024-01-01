import os
import urllib.request as request
from zipfile import ZipFile
from pathlib import Path
from objectClassifier.entity.config_entity import PrepareBaseModelConfig

import tensorflow as tf
from keras.models import Model
from tensorflow.keras.applications import Xception
from keras.layers import Flatten, Dense, Dropout, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        self.model = Xception(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def prepare_full_model(self, model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False

        head = model.output
        head = MaxPooling2D(pool_size=(2,2))(head)
        head = Flatten()(head)
        head = Dense(256, activation='relu')(head)
        head = Dropout(0.5)(head)
        head = Dense(classes, activation='softmax')(head)

        model = Model(inputs=model.input, outputs=head)

        lr_schedule = ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=self.config.params_decay_steps,
            decay_rate=self.config.params_decay_rate)
        optimizer = Adam(learning_rate=lr_schedule)
        model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])

        return model
    
    def update_base_model(self):
        self.full_model = self.prepare_full_model(self,
            model = self.model,
            classes = self.config.params_classes,
            freeze_all = True,
            freeze_till = None,
            learning_rate = self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: Model):
        model.save(path)

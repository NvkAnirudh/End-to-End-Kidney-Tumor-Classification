import os
from objectClassifier.constants import *
from objectClassifier.utils.helper import read_yaml, create_directories
from objectClassifier.entity.config_entity import DataIngestionConfig, PrepareBaseModelConfig, TrainingConfig, EvaluationConfig

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            zip_data_file=config.zip_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    def get_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.base_model

        create_directories([config.root_dir])

        base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.image_size,
            params_learning_rate=self.params.learning_rate,
            params_include_top=self.params.include_top,
            params_weights=self.params.weights,
            params_classes=self.params.classes,
            params_decay_steps=self.params.decay_steps,
            params_decay_rate=self.params.decay_rate
        )

        return base_model_config
    
    def get_training_config(self) -> TrainingConfig:
        config = self.config.training
        base_model = self.config.base_model

        training_data = os.path.join(self.config.data_ingestion.unzip_dir,'train')
        validation_data = os.path.join(self.config.data_ingestion.unzip_dir,'valid')
        create_directories([config.root_dir])

        training_config = TrainingConfig(
            root_dir=Path(config.root_dir),
            trained_model_path=Path(config.trained_model_path),
            updated_base_model_path=Path(base_model.updated_base_model_path),
            training_data=Path(training_data),
            validation_data=Path(validation_data),
            params_epochs=self.params.epochs,
            params_batch_size=self.params.batch_size,
            params_is_augmentation=self.params.augmentation,
            params_image_size=self.params.image_size
        )

        return training_config
    
    def get_evaluation_config(self) -> EvaluationConfig:
        evaluation_config = EvaluationConfig(
            path_of_model="artifacts/training/model.h5",
            testing_data="artifacts/data_ingestion/test",
            mlflow_uri="https://dagshub.com/NvkAnirudh/End-to-End-Kidney-Tumor-Classification.mlflow",
            all_params=self.params,
            params_image_size=self.params.image_size,
            params_batch_size=self.params.batch_size
        )

        return evaluation_config
        

from objectClassifier.config.configuration import ConfigurationManager
from objectClassifier.components.model_training import Training
from objectClassifier import logger

stage_name = 'training'

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.data_generator()
        training.train()
        
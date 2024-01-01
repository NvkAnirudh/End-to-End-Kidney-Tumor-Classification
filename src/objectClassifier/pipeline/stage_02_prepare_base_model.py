from objectClassifier.config.configuration import ConfigurationManager
from objectClassifier.components.prepare_base_model import PrepareBaseModel
from objectClassifier import logger

stage_name = 'Preparation of Base Model'

class PrepareBaseModelPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        base_model_config = config.get_base_model_config()
        base_model = PrepareBaseModel(config=base_model_config)
        base_model.get_base_model()
        base_model.update_base_model()
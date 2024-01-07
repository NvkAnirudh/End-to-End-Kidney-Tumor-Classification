from objectClassifier.config.configuration import ConfigurationManager
from objectClassifier.components.model_evaluation_mlflow import Evaluation
from objectClassifier import logger

stage_name = 'Evaluation stage'

class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        evaluation_config = config.get_evaluation_config()
        evaluation = Evaluation(config=evaluation_config)
        evaluation.evaluation()
        evaluation.log_into_mlflow()
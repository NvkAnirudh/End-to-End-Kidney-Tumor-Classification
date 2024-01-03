from objectClassifier import logger
from objectClassifier.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from objectClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelPipeline
from objectClassifier.pipeline.stage_03_model_training import ModelTrainingPipeline
from objectClassifier.pipeline.stage_04_model_evaluation_mlflow import EvaluationPipeline

stage_name = 'Data Ingestion Stage'
try:
    logger.info(f'{stage_name} started')
    pipeline = DataIngestionPipeline()
    pipeline.main()
    logger.info(f'{stage_name} completed')
except Exception as e:
    logger.exception(e)
    raise e

stage_name = 'Preparation of Base Model'
try:
    logger.info(f'{stage_name} started')
    pipeline = PrepareBaseModelPipeline()
    pipeline.main()
    logger.info(f'{stage_name} completed')
except Exception as e:
    logger.exception(e)
    raise e

stage_name = 'Training'
try:
    logger.info(f'{stage_name} started')
    pipeline = ModelTrainingPipeline()
    pipeline.main()
    logger.info(f'{stage_name} completed')
except Exception as e:
    logger.exception(e)
    raise e

stage_name = 'Evaluation'
try:
    logger.info(f'{stage_name} started')
    pipeline = EvaluationPipeline()
    pipeline.main()
    logger.info(f'{stage_name} completed')
except Exception as e:
    logger.exception(e)
    raise e
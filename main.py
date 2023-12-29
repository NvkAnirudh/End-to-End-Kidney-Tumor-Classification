from objectClassifier import logger
from objectClassifier.pipeline.stage_01_data_ingestion import DataIngestionPipeline

stage_name = 'Data Ingestion Stage'
try:
    logger.info(f'{stage_name} started')
    pipeline = DataIngestionPipeline()
    pipeline.main()
    logger.info('f{stage_name} completed \n\n')
except Exception as e:
    logger.exception(e)
    raise e
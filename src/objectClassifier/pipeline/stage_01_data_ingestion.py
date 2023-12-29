from objectClassifier.config.configuration import ConfigurationManager
from objectClassifier.components.data_ingestion import DataIngestion
from objectClassifier import logger

stage_name = 'Data Ingestion Stage'

class DataIngestionPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_data()
        data_ingestion.extract_zip_file()

if __name__=='__main__':
    try:
        logger.info(f'{stage_name} started')
        pipeline = DataIngestionPipeline()
        pipeline.main()
        logger.info(f'{stage_name} completed \n\n')
    except Exception as e:
        logger.exception(e)
        raise e
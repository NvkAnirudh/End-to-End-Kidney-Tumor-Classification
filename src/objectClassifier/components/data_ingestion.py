import os
import zipfile
import gdown
from objectClassifier import logger
from objectClassifier.utils.helper import get_size
from objectClassifier.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_data(self) -> str:
        # Fetching data from a URL

        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.zip_data_file

            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info('Downloading data from {dataset_url} into {zip_download_dir}')

            file_id = dataset_url.split('/')[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix+file_id, zip_download_dir)

            logger.info('Downloaded data from {dataset_url} into {zip_download_dir}')

        except Exception as e:
            raise e
        
    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.zip_data_file,'r') as zip_ref:
            zip_ref.extractall(unzip_path)
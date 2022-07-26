
import sys
import  os
import pandas as pd

from IMDB.app_entity.config_entity import DataIngestionConfig, DataValidationConfig, \
    TrainingPipelineConfig, DataTransformationConfig, ModelTrainerConfig, ModelPusherConfig, ModelEvaluationConfig
from IMDB.app_exception.exception import App_Exception
from IMDB.app_logger import App_Logger
from IMDB.app_util.util import read_yaml_file
from IMDB.app_constants import *

logging = App_Logger(__name__)


class Configuration:

    def __init__(self,
                 config_file_path: str = CONFIG_FILE_PATH) -> None:
        try:
            self.config_info = read_yaml_file(file_path=config_file_path)
            self.pipeline_config = self.get_training_pipeline_config()
            self.time_stamp = CURRENT_TIME_STAMP

        except Exception as e:
            raise App_Exception(e, sys) from e

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            logging.info(f"Config file path: {self.config_info}")
            logging.info(f"Loading data ingestion config")
            artifact_dir = self.pipeline_config.artifact_dir
            data_ingestion_config_info = self.config_info[DATA_INGESTION_CONFIG_KEY]
            data_download_url = data_ingestion_config_info[DATA_INGESTION_DATA_KEY]
            data_download_file_name = data_ingestion_config_info[DATA_INGESTION_DOWNLOAD_FILE_NAME_KEY]
            data_ingestion_dir = data_ingestion_config_info[DATA_INGESTION_DIR_KEY]
            raw_data_dir_name = data_ingestion_config_info[DATA_INGESTION_RAW_DIR_KEY]
            raw_data_file_name = data_ingestion_config_info[DATA_INGESTION_RAW_DATA_FILE_NAME_KEY]
            ingested_data_dir_name = data_ingestion_config_info[DATA_INGESTION_INGESTED_DIR_KEY]
            ingested_train_filename = data_ingestion_config_info[DATA_INGESTION_INGESTED_TRAIN_FILE_NAME_KEY]
            ingested_test_dir = data_ingestion_config_info[DATA_INGESTION_INGESTED_TEST_FILE_NAME_KEY]
            ingested_train_collection = data_ingestion_config_info[DATA_INGESTION_INGESTED_TRAIN_COLLECTION_KEY]
            ingested_test_collection = data_ingestion_config_info[DATA_INGESTION_INGESTED_TEST_COLLECTION_KEY]
            
            raw_data_file_path = os.path.join(artifact_dir, data_ingestion_dir, raw_data_dir_name, data_download_file_name)
            raw_data_file_path_to_ingest = os.path.join(os.path.dirname(raw_data_file_path), raw_data_file_name)
            ingested_train_file_path = os.path.join(artifact_dir, data_ingestion_dir, ingested_data_dir_name, ingested_train_filename)
            ingested_test_file_path = os.path.join(artifact_dir, data_ingestion_dir, ingested_data_dir_name, ingested_test_dir)  
            
            os.makedirs(os.path.dirname(raw_data_file_path_to_ingest), exist_ok=True)
            os.makedirs(os.path.dirname(ingested_test_file_path), exist_ok=True)

            data_ingestion_config = DataIngestionConfig(dataset_download_url=data_download_url,
                                                        dataset_download_file_name = data_download_file_name,
                                                        raw_data_file_path=raw_data_file_path,
                                                        raw_file_path_to_ingest=raw_data_file_path_to_ingest,
                                                        ingested_train_file_path=ingested_train_file_path,
                                                        ingested_test_data_path=ingested_test_file_path,
                                                        ingested_train_collection=ingested_train_collection,
                                                        ingested_test_collection=ingested_test_collection)
            logging.info(f"Data ingestion config: {data_ingestion_config}")
            return data_ingestion_config
            
            
        except Exception as e:
            raise App_Exception(e, sys) from e

    def get_training_pipeline_config(self) -> TrainingPipelineConfig:
        try:
            training_pipeline_config = self.config_info[TRAINING_PIPELINE_CONFIG_KEY]
            pipeline_dir = training_pipeline_config[TRAINING_PIPELINE_NAME_KEY]
            pipeline_artifact_dir = training_pipeline_config[TRAINING_PIPELINE_ARTIFACT_DIR_KEY]
            artifact_dir = os.path.join(ROOT_DIR,pipeline_dir , pipeline_artifact_dir)
            os.makedirs(artifact_dir, exist_ok=True)
            training_pipeline_config = TrainingPipelineConfig(artifact_dir=artifact_dir,
                                                              pipeline_name=pipeline_dir)
            logging.info(f"Training pipeline config: {training_pipeline_config}")
            return training_pipeline_config
        except Exception as e:
            raise App_Exception(e, sys) from e

    def get_data_validation_config(self) -> DataValidationConfig:
        try:
            pass
        except Exception as e:
            raise App_Exception(e, sys) from e

    def get_data_transformation_config(self) -> DataTransformationConfig:
        try:
            pass 
        except Exception as e:
            raise App_Exception(e, sys) from e

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        try:
            pass 
        except Exception as e:
            raise App_Exception(e, sys) from e

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        try:
            pass 
        except Exception as e:
            raise App_Exception(e, sys) from e

    def get_model_pusher_config(self) -> ModelPusherConfig:
        try:
            pass 
        except Exception as e:
            raise App_Exception(e, sys) from e

from pyexpat import model
from statistics import mode
import sys
import  os
import pandas as pd

from IMDB.app_entity.config_entity import DataIngestionConfig, TrainingPipelineConfig, DataTransformationConfig, ModelTrainerConfig,ModelEvaluationConfig
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

        
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        try:
            artifact_dir = self.pipeline_config.artifact_dir
            data_transformation_config_info = self.config_info[DATA_TRANSFORMATION_CONFIG_KEY]
            data_ingestion_config_info = self.config_info[DATA_INGESTION_CONFIG_KEY]
            data_transformation_dir = data_transformation_config_info[DATA_TRANSFORMATION_DIR_KEY]
            transformed_dir = data_transformation_config_info[DATA_TRANSFORMATION_DIR_NAME_KEY]
            transformed_train_dir_name = data_transformation_config_info[DATA_TRANSFORMATION_TRAIN_DIR_NAME_KEY]
            transformed_test_dir_name = data_transformation_config_info[DATA_TRANSFORMATION_TEST_DIR_NAME_KEY]
            preprocessing_dir = data_transformation_config_info[DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY]
            preprocessed_obj_name = data_transformation_config_info[DATA_TRANSFORMATION_PREPROCESSED_FILE_NAME_KEY]
            ingested_train_collection = data_ingestion_config_info[DATA_INGESTION_INGESTED_TRAIN_COLLECTION_KEY]
            ingested_test_collection = data_ingestion_config_info[DATA_INGESTION_INGESTED_TEST_COLLECTION_KEY]
            
            transformed_train_dir = os.path.join(artifact_dir, data_transformation_dir, transformed_dir, transformed_train_dir_name , "train.npz")
            transformed_test_dir = os.path.join(artifact_dir, data_transformation_dir, transformed_dir, transformed_test_dir_name, "test.npz")
            preprocessed_obj_path = os.path.join(artifact_dir, data_transformation_dir, preprocessing_dir, preprocessed_obj_name)
            
            os.makedirs(os.path.dirname(transformed_train_dir), exist_ok=True)
            os.makedirs(os.path.dirname(transformed_test_dir), exist_ok=True)

            data_transformation_config = DataTransformationConfig(transformed_train_file_path = transformed_train_dir,
                                                                   transformed_test_file_path = transformed_test_dir,
                                                                   preprocessed_object_file_path = preprocessed_obj_path,
                                                                   ingested_train_collection= ingested_train_collection,
                                                                   ingested_test_collection= ingested_test_collection)
            return data_transformation_config
            
        except Exception as e:
            raise App_Exception(e, sys) from e

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        
        try:
            artifact_dir = self.pipeline_config.artifact_dir
            data_transformation_config = self.get_data_transformation_config()
            model_trainer_config_info = self.config_info[MODEL_TRAINER_CONFIG_KEY]
            
            model_trainer_dir = model_trainer_config_info[MODEL_TRAINER_ARTIFACT_DIR]
            model_config_dir = model_trainer_config_info[MODEL_TRAINER_MODEL_CONFIG_DIR_KEY]
            model_config_file_name = model_trainer_config_info[MODEL_TRAINER_MODEL_CONFIG_FILE_NAME_KEY]
            base_accuracy = model_trainer_config_info[MODEL_TRAINER_BASE_ACCURACY_KEY]
            trained_model_dir_name = model_trainer_config_info[MODEL_TRAINER_TRAINED_MODEL_DIR_KEY]
            trained_model_file_name = model_trainer_config_info[MODEL_TRAINER_TRAINED_MODEL_FILE_NAME_KEY]
            
            transformed_train_file_path = data_transformation_config.transformed_train_file_path
            transformed_test_file_path = data_transformation_config.transformed_test_file_path
            preprocessed_object_file_path=data_transformation_config.preprocessed_object_file_path
            
            model_config_file_path = os.path.join(model_config_dir, model_config_file_name)
            model_trainer_artifact_dir = os.path.join(
                artifact_dir, model_trainer_dir)

            trained_model_file_path = os.path.join(model_trainer_artifact_dir, trained_model_dir_name,
                                                   trained_model_file_name)
            
            
        

            model_config_file_path = os.path.join(model_config_dir, model_config_file_name)
            os.makedirs(os.path.dirname(trained_model_file_path), exist_ok=True)


            model_trainer_config = ModelTrainerConfig(transformed_train_file_path= transformed_train_file_path,
                                                      transformed_test_file_path= transformed_test_file_path,
                                                      preprocessed_object_file_path= preprocessed_object_file_path,
                                                      model_config_file_path= model_config_file_path,
                                                      base_accuracy= base_accuracy,
                                                      trained_model_file_path= trained_model_file_path)
    
            logging.info(f"Model trainer config: {model_trainer_config}")
            return model_trainer_config
        except Exception as e:
            raise App_Exception(e, sys) from e

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        try:
            artifact_dir = self.pipeline_config.artifact_dir
            model_evaluation_config_info = self.config_info[MODEL_EVALUATION_CONFIG_KEY]
            data_ingestion_config = self.get_data_ingestion_config()
            model_trainer_config = self.get_model_trainer_config()
            model_evaluation_dir_name = model_evaluation_config_info[MODEL_EVALUATION_ARTIFACT_DIR]
            model_evaluation_dir = os.path.join(artifact_dir, model_evaluation_dir_name)
            os.makedirs(model_evaluation_dir, exist_ok=True)
            timestamp = CURRENT_TIME_STAMP
            model_name = model_trainer_config.trained_model_file_path.split("/")[-1]
            model_evaluated_file_path = os.path.join(artifact_dir, model_evaluation_dir, timestamp,model_name)
            
            model_evaluation_collection = data_ingestion_config.ingested_test_collection
            trained_model_path = model_trainer_config.trained_model_file_path
    
            response = ModelEvaluationConfig(model_evaluation_collection=model_evaluation_collection,
                                              trained_model_path=trained_model_path,
                                              model_evaluated_file_path=model_evaluated_file_path ,
                                              model_evaluation_dir=model_evaluation_dir)
            return response
        except Exception as e:
            raise App_Exception(e, sys) from e

    # def get_model_pusher_config(self) -> ModelPusherConfig:
    #     try:
    #         pass 
    #     except Exception as e:
    #         raise App_Exception(e, sys) from e
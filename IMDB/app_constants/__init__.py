import os
from datetime import datetime


def get_current_time_stamp():
    return f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"


ROOT_DIR = os.getcwd()  # to get current working directory
CURRENT_TIME_STAMP = get_current_time_stamp()


# config constants
CONFIG_DIR = os.path.join(ROOT_DIR, 'config')
CONFIG_FILE_NAME = "config.yaml"
CONFIG_FILE_PATH = os.path.join(CONFIG_DIR, CONFIG_FILE_NAME)

# Training pipeline related variable
TRAINING_PIPELINE_CONFIG_KEY = "training_pipeline_config"
TRAINING_PIPELINE_ARTIFACT_DIR_KEY = "artifact_dir"
TRAINING_PIPELINE_NAME_KEY = "pipeline_name"


#   data_ingestion_dir : stage00_data_ingestion
#   dataset_download_url: lakshmi25npathi/imdb-dataset-of-50k-movie-reviews 
#   dataset_download_file_name : imdb-dataset-of-50k-movie-reviews.zip
#   raw_data_dir: raw_data
#   raw_data_file_name: IMDB Dataset.csv
#   ingested_dir: ingested_data
#   ingested_data_Train_file_name: Train.csv
#   ingested_data_Test_file_name: Test.csv
#   ingested_data_Train_collection_name: ingested_train
#   ingested_data_Test_collection_name: ingested_test

# Data Ingestion related variable
DATA_INGESTION_CONFIG_KEY = "data_ingestion_config"
DATA_INGESTION_DIR_KEY = "data_ingestion_dir"
DATA_INGESTION_DATA_KEY = 'dataset_download_url'
DATA_INGESTION_DOWNLOAD_FILE_NAME_KEY = 'dataset_download_file_name'
DATA_INGESTION_INGESTED_DIR_KEY = "ingested_dir"
DATA_INGESTION_RAW_DIR_KEY = "raw_data_dir"
DATA_INGESTION_RAW_DATA_FILE_NAME_KEY = "raw_data_file_name"
DATA_INGESTION_INGESTED_TRAIN_FILE_NAME_KEY = "ingested_data_Train_file_name"
DATA_INGESTION_INGESTED_TEST_FILE_NAME_KEY = "ingested_data_Test_file_name"
DATA_INGESTION_INGESTED_TRAIN_COLLECTION_KEY = "ingested_data_Train_collection_name"
DATA_INGESTION_INGESTED_TEST_COLLECTION_KEY = "ingested_data_Test_collection_name" 


# Data Transformation related variables
DATA_TRANSFORMATION_CONFIG_KEY = "data_transformation_config"
DATA_TRANSFORMATION_DIR_KEY = "data_transformation_dir"
DATA_TRANSFORMATION_DIR_NAME_KEY = "transformed_dir"
DATA_TRANSFORMATION_TRAIN_DIR_NAME_KEY = "transformed_train_dir"
DATA_TRANSFORMATION_TEST_DIR_NAME_KEY = "transformed_test_dir"
DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY = "preprocessing_dir"
DATA_TRANSFORMATION_PREPROCESSED_FILE_NAME_KEY = "preprocessed_object_file_name"


# Model Training related variables

MODEL_TRAINER_ARTIFACT_DIR = "model_trainer_dir"
MODEL_TRAINER_CONFIG_KEY = "model_trainer_config"
MODEL_TRAINER_TRAINED_MODEL_DIR_KEY = "trained_model_dir"
MODEL_TRAINER_TRAINED_MODEL_FILE_NAME_KEY = "model_file_name"
MODEL_TRAINER_BASE_ACCURACY_KEY = "base_accuracy"
MODEL_TRAINER_MODEL_CONFIG_DIR_KEY = "model_config_dir"
MODEL_TRAINER_MODEL_CONFIG_FILE_NAME_KEY = "model_config_file_name"



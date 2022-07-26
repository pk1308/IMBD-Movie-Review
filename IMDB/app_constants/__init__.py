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

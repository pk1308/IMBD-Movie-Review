
import shutil
import sys, os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import kaggle 
import subprocess
from zipfile import ZipFile

from IMDB.app_config.configuration import Configuration
from IMDB.app_entity.config_entity import DataIngestionConfig
from IMDB.app_entity.artifacts_entity import DataIngestionArtifact
from IMDB.app_exception.exception import App_Exception
from IMDB.app_logger import App_Logger
from IMDB.app_database.mongoDB import MongoDB

logging = App_Logger(__name__)


class DataIngestion:

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            logging.info(f"{'>>' * 20}Data Ingestion log started.{'<<' * 20} ")
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            raise App_Exception(e, sys)

    def download_data(self, dataset_download_url : str, dataset_download_file_name :str,
                      raw_data_file_path : str) -> str:
        try:
            # extraction remote url to download dataset
            dataset_download_url = dataset_download_url
            dataset_download_file_name = dataset_download_file_name
            raw_data_file_path = raw_data_file_path
            
            kaggle.api.authenticate()
            subprocess.run(['kaggle', 'datasets', 'download', '-d', dataset_download_url])
            shutil.move(dataset_download_file_name, raw_data_file_path)
            with ZipFile(raw_data_file_path, 'r') as zipObj:
            
                zipObj.extractall(os.path.dirname(raw_data_file_path))
            

            return True

        except Exception as e:
            raise App_Exception(e, sys) from e

    def split_data_as_train_test(self) -> DataIngestionArtifact:
        try:
            raw_data_file_path = self.data_ingestion_config.raw_file_path_to_ingest

            logging.info(f"Reading csv file: [{raw_data_file_path}]")
            raw_data_frame = pd.read_csv(raw_data_file_path)

            logging.info(f"Splitting data into train and test")
            strat_train_set = None
            strat_test_set = None

            split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

            for train_index, test_index in split.split(raw_data_frame, raw_data_frame["sentiment"]):
                strat_train_set = raw_data_frame.loc[train_index]
                strat_test_set = raw_data_frame.loc[test_index]

            train_file_path = self.data_ingestion_config.ingested_train_file_path

            test_file_path = self.data_ingestion_config.ingested_test_data_path
            
            if strat_train_set is not None:
                logging.info(f"Exporting training dataset to file: [{train_file_path}]")
                strat_train_set.to_csv(train_file_path, index=False)
                train_collection_name = self.data_ingestion_config.ingested_train_collection
                logging.info(f"Exporting training dataset to MongoDB: [{train_collection_name}]")
                train_conn = MongoDB(train_collection_name , drop_collection=True)
                status = train_conn.Insert_Many(strat_train_set.to_dict('records'))
                if status is True:
                    logging.info(f"Training dataset exported to MongoDB: [{train_collection_name}]")
                

            if strat_test_set is not None:
                logging.info(f"Exporting test dataset to file: [{test_file_path}]")
                strat_test_set.to_csv(test_file_path, index=False)
                test_collection_name = self.data_ingestion_config.ingested_test_collection
                test_conn = MongoDB(test_collection_name , drop_collection=True)
                status = test_conn.Insert_Many(strat_test_set.to_dict('records'))
                if status is True:
                    logging.info(f"Test dataset exported to MongoDB: [{test_collection_name}]")
                

            data_ingestion_artifact = DataIngestionArtifact(train_file_path=train_file_path,
                                                            test_file_path=test_file_path,
                                                            is_ingested=True,
                                                            message=f"Data ingestion completed successfully."
                                                            )
            logging.info(f"Data Ingestion artifact:[{data_ingestion_artifact}]")
            return data_ingestion_artifact

        except Exception as e:
            raise App_Exception(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion_config = self.data_ingestion_config
            dataset_download_url = data_ingestion_config.dataset_download_url
            dataset_download_file_name  = data_ingestion_config.dataset_download_file_name
            raw_data_file_path = data_ingestion_config.raw_data_file_path
            self.download_data(dataset_download_url, dataset_download_file_name , raw_data_file_path)

            data_ingestion_response = self.split_data_as_train_test()
            return data_ingestion_response
        except Exception as e:
            raise App_Exception(e, sys) from e

    def __del__(self):
        logging.info(f"{'>>' * 20}Data Ingestion log completed.{'<<' * 20} \n\n")
        
if __name__ == "__main__":
    config = Configuration()
    data_ingestion_config = config.get_data_ingestion_config()
    data_ingestion = DataIngestion(data_ingestion_config)
    data_ingestion_response = data_ingestion.initiate_data_ingestion()
    logging.info(f"Data Ingestion artifact:[{data_ingestion_response}]")
    logging.info(f"{'>>' * 20}Data Ingestion log completed.{'<<' * 20} \n\n")

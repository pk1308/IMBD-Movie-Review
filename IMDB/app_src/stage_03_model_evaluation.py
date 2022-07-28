
import pandas as pd
import numpy as np
import glob
from sklearn import preprocessing
from bs4 import BeautifulSoup
import re
import nltk
from nltk.tokenize.toktok import ToktokTokenizer

from IMDB.app_exception.exception import App_Exception
from IMDB.app_logger import App_Logger
from IMDB.app_entity.config_entity import ModelEvaluationConfig
from IMDB.app_entity.artifacts_entity import ModelEvaluationArtifact
from IMDB.app_constants import *
from IMDB.app_util.util import save_object, load_object , load_data_from_mongodb
from IMDB.app_entity.model_factory import evaluate_classification_model
from IMDB.app_config.configuration import Configuration
from IMDB.app_database.mongoDB import MongoDB
import os
import sys


logging = App_Logger(__name__)

class ModelEvaluation:

    def __init__(self, model_evaluation_config: ModelEvaluationConfig):
        try:
            logging.info(f"{'>>' * 30}Model Evaluation log started.{'<<' * 30} ")
            self.model_evaluation_config = model_evaluation_config
        except Exception as e:
            raise App_Exception(e, sys) from e

    def get_best_model(self):
        try:
            model = None
            model_dir = self.model_evaluation_config.model_evaluation_dir
            if not os.listdir(model_dir):
                logging.info("No model found in model evaluation directory")
                return model
            list_of_files = glob.glob(f"{model_dir}/*") 
            latest_model_dir = max(list_of_files, key=os.path.getctime)
            file_name = "model.pkl"
            latest_model_path = os.path.join(model_dir,latest_model_dir, file_name)
            if os.path.isfile(latest_model_dir ):
                model = load_object(file_path=latest_model_path)
                return model
            else:
                return None
        except Exception as e:
            raise App_Exception(e, sys) from e


    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
    
            model_evaluation_config_info = self.model_evaluation_config
            model_evaluation_collection_test = model_evaluation_config_info.model_evaluation_collection
            trained_model_path = model_evaluation_config_info.trained_model_path
            model_evaluated_file_path = model_evaluation_config_info.model_evaluated_file_path
            model_evaluation_collection_train = "ingested_train"
    
            train_conn = MongoDB(collection_name=model_evaluation_collection_train , drop_collection=False)
            train_df = load_data_from_mongodb(train_conn, limit=40000)
            train_X = train_df.drop(columns=['sentiment'])
            train_y = train_df['sentiment']
            test_conn = MongoDB(collection_name=model_evaluation_collection_test , drop_collection=False)
            test_df = load_data_from_mongodb(test_conn , limit=40000)
            test_X = test_df.drop(columns=['sentiment'])
            test_y = test_df['sentiment']
            
            model = self.get_best_model()
            trained_model = load_object(file_path=trained_model_path)
            trained_model_object = trained_model.trained_model_object
            
            if model is None:
                os.makedirs(os.path.dirname(model_evaluated_file_path), exist_ok=True)
                save_object(obj=trained_model, file_path=model_evaluated_file_path)
                logging.info("Not found any existing model. Hence accepting trained model")
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=model_evaluated_file_path,
                                                                    is_model_accepted=True)
                logging.info(f"Model accepted. Model eval artifact {model_evaluation_artifact} created")
                return model_evaluation_artifact

            model_list = [model, trained_model_object]
            preprocessed_test = trained_model.preprocessing_object.transform(test_X)
            test_label = trained_model.label_binarizer_object.transform(test_y)
            preprocessed_train = trained_model.preprocessing_object.transform(train_X)
            train_label = trained_model.label_binarizer_object.transform(train_y)

            metric_info_artifact = evaluate_classification_model(model_list=model_list,
                                                             X_train=preprocessed_train,
                                                             y_train=train_label,
                                                             X_test=preprocessed_test,
                                                             y_test=test_label,
                                                             base_accuracy=self.model_trainer_artifact.model_accuracy,
                                                             )
            logging.info(f"Model evaluation completed. model metric artifact: {metric_info_artifact}")

            if metric_info_artifact is None:
                response = ModelEvaluationArtifact(is_model_accepted=False,
                                                   evaluated_model_path=None)
                logging.info(response)
                return response

            if metric_info_artifact.index_number == 1:
                save_object(obj =trained_model, file_path=model_evaluated_file_path)
                os.makedirs(os.path.dirname(model_evaluated_file_path), exist_ok=True)
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=model_evaluated_file_path,
                                                                    is_model_accepted=True)
                logging.info(f"Model accepted. Model eval artifact {model_evaluation_artifact} created")

            else:
                logging.info("Trained model is no better than existing model hence not accepting trained model")
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=False,
                                                                    is_model_accepted=False)
            return model_evaluation_artifact
        except Exception as e:
            raise App_Exception(e, sys) from e

    def __del__(self):
        logging.info(f"{'=' * 20}Model Evaluation log completed.{'=' * 20} ")
        
if __name__ == "__main__":
   config = Configuration()
   model_evaluation_config = config.get_model_evaluation_config()
   model_evaluation = ModelEvaluation(model_evaluation_config)
   
   model_evaluation_artifact= model_evaluation.initiate_model_evaluation()
   
   logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
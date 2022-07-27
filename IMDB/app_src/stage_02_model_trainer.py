
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
from nltk.tokenize.toktok import ToktokTokenizer

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from IMDB.app_config.configuration import Configuration
from IMDB.app_exception.exception import App_Exception
from IMDB.app_logger import App_Logger
from IMDB.app_entity.artifacts_entity import DataTransformationArtifact, ModelTrainerArtifact
from IMDB.app_entity.config_entity import ModelTrainerConfig
from IMDB.app_database.mongoDB import MongoDB
from IMDB.app_entity.model_factory import MetricInfoArtifact, ModelFactory, GridSearchedBestModel, evaluate_classification_model
from IMDB.app_util.util import load_numpy_array_data, save_object, load_object , load_data_from_mongodb


import sys
from typing import List

logging = App_Logger(__name__)


class EstimatorModel:
    def __init__(self, preprocessing_object, trained_model_object , label_binarizer_object):
        """
        TrainedModel constructor
        preprocessing_object: preprocessing_object
        trained_model_object: trained_model_object
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object
        self.label_binarizer_object = label_binarizer_object

    def predict(self, X):
        """
        function accepts raw inputs and then transformed raw input using preprocessing_object
        which guarantees that the inputs are in the same format as the training data
        At last it perform prediction on transformed features
        """
        transformed_feature = self.preprocessing_object.transform(X)
        predicted = self.trained_model_object.predict(transformed_feature)
        return self.label_binarizer_object.inverse_transform(predicted)

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"


class ModelTrainer:

    def __init__(self, model_trainer_config: ModelTrainerConfig):
        try:
            logging.info(f"{'>>' * 30}Model trainer log started.{'<<' * 30} ")
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise App_Exception(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            model_trainer_config = self.model_trainer_config
            # transformed_train_file_path = model_trainer_config.transformed_train_file_path
            # transformed_test_file_path = model_trainer_config.transformed_test_file_path
            model_config_file_path = model_trainer_config.model_config_file_path
            base_accuracy = model_trainer_config.base_accuracy
            preprocessed_object_file_path=model_trainer_config.preprocessed_object_file_path
            trained_model_file_path = model_trainer_config.trained_model_file_path
            
            train_collection_name = "ingested_train"
            train_conn = MongoDB(train_collection_name , drop_collection=False)


            logging.info(f"Loading training and test data as pandas dataframe.")
            train_df = load_data_from_mongodb(train_conn , limit=40000)


            target_column_name = "sentiment"

            logging.info(f"Splitting input and target feature from training and testing dataframe.")
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            logging.info(f"input :{input_feature_train_df.shape}")
            logging.info(f"lable : {target_feature_train_df.shape}")

            label_binarizer_object = LabelBinarizer()
            target_feature = label_binarizer_object.fit_transform(target_feature_train_df)
            
            preprocessing_obj = load_object(file_path=preprocessed_object_file_path)
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)


            logging.info(f"Splitting training and testing input and target feature")
            X_train, X_test, y_train, y_test = train_test_split(input_feature_train_arr, target_feature.ravel(),test_size=0.2, random_state=42)

            logging.info(f"Extracting model config file path")

            logging.info(f"Initializing model factory class using above model config file: {model_config_file_path}")
            model_factory = ModelFactory(model_config_path=model_config_file_path)
            logging.info(f"Expected accuracy: {base_accuracy}")

            logging.info(f"Initiating operation model selection")
            best_model = model_factory.get_best_model(X=X_train, y=y_train, base_accuracy=base_accuracy)

            logging.info(f"Best model found on training dataset: {best_model}")

            logging.info(f"Extracting trained model list.")
            grid_searched_best_model_list: List[GridSearchedBestModel] = model_factory.grid_searched_best_model_list

            model_list = [model.best_model for model in grid_searched_best_model_list]
            logging.info(f"Model list: {model_list} , {len(model_list)}")
    
            logging.info(f"Evaluation all trained model on training and testing dataset both")
            metric_info: MetricInfoArtifact = evaluate_classification_model(model_list=model_list, X_train=X_train,
                                                                        y_train=y_train, X_test=X_test, y_test=y_test,
                                                                        base_accuracy=base_accuracy)
            if metric_info.model_object is None:
                raise App_Exception("Best model not found")

            logging.info(f"Best found model on both training and testing dataset.")


            model_object = metric_info.model_object

        
            trained_model_estimator = EstimatorModel(preprocessing_object=preprocessing_obj, 
                                                     trained_model_object=model_object ,
                                                     label_binarizer_object=label_binarizer_object)
            logging.info(f"Saving model at path: {trained_model_file_path}")
            save_object(file_path=trained_model_file_path, obj=trained_model_estimator)

            model_trainer_artifact = ModelTrainerArtifact(is_trained=True, message="Model Trained successfully",
                                                          trained_model_file_path=trained_model_file_path,
                                                          train_f1=metric_info.train_f1,
                                                          test_f1=metric_info.test_f1,
                                                          train_precision=metric_info.train_precision,
                                                          test_precision=metric_info.test_precision,
                                                          model_accuracy=metric_info.model_accuracy

                                                          )

            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise App_Exception(e, sys) from e

    def __del__(self):
        logging.info(f"{'>>' * 30}Model trainer log completed.{'<<' * 30} ")
        
        
if __name__ == "__main__":
    
    config = Configuration()
    model_trainer_config = config.get_model_trainer_config()
    model_trainer = ModelTrainer(model_trainer_config)
    model_trainer_artifact = model_trainer.initiate_model_trainer()
    

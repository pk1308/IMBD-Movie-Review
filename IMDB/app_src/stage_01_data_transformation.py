
import sys
from nltk import data
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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

from IMDB.app_entity.artifacts_entity import DataIngestionArtifact, DataTransformationArtifact
from IMDB.app_entity.config_entity import DataTransformationConfig
from IMDB.app_logger import App_Logger
from IMDB.app_config.configuration import Configuration
from IMDB.app_database.mongoDB import MongoDB
from IMDB.app_exception.exception import App_Exception
from IMDB.app_util.util import save_object, save_numpy_array_data , load_data_from_mongodb
from IMDB.app_constants import *

logging = App_Logger(__name__)

class FeatureGenerator(BaseEstimator, TransformerMixin):

    def __init__(self):
        try:
            pass
        except Exception as e:
            raise App_Exception(e, sys) from e

    def fit(self, X, y=None):
        pass 
        return self

    def transform(self, X, y=None):
        try:
            logging.info("Transforming data")
            data = X.copy()
            review_column = 'review'
            data = data.apply(self.strip_html)
            data = data.apply(self.remove_between_square_brackets)
            data = data.apply(self.remove_special_characters)
            data= data.apply(self.simple_stemmer)
            data= data.apply(self.remove_stopwords)
            return data
        except Exception as e:
            raise App_Exception(e, sys) from e
    
    def strip_html(self , text):
        try:
            soup = BeautifulSoup(text, 'html.parser')
            return soup.get_text()
        except Exception as e:
            raise App_Exception(e, sys) from e
        
    def remove_between_square_brackets(self,text):
        try:
            return re.sub('\[[^]]*\]', '', text)
        except Exception as e:
            raise App_Exception(e, sys) from e
        
    def remove_special_characters(self ,text, remove_digits=True):
        try:
            pattern=r'[^a-zA-z0-9\s]'
            text=re.sub(pattern,'',text)
            return text
        except Exception as e:
            raise App_Exception(e, sys) from e
        
    def simple_stemmer(self,text):
        try : 
            ps=nltk.porter.PorterStemmer()
            text= ' '.join([ps.stem(word) for word in text.split()])
            return text
        except Exception as e:
            raise App_Exception(e, sys) from e
        
 
    def remove_stopwords(self,text, is_lower_case=False):
        tokenizer=ToktokTokenizer()
        stopword_list=set(nltk.corpus.stopwords.words('english'))
        tokens = tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        if is_lower_case:
            filtered_tokens = [token for token in tokens if token not in stopword_list]
        else:
            filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
        filtered_text = ' '.join(filtered_tokens)    
        return filtered_text
        


class DataTransformation:

    def __init__(self, data_transformation_config: DataTransformationConfig):
        try:
            logging.info(f"{'>>' * 30}Data Transformation log started.{'<<' * 30} ")
            self.data_transformation_config = data_transformation_config


        except Exception as e:
            raise App_Exception(e, sys) from e
    def get_preprocessing_obj(self):
        try:
            review_column = 'review'
            feature_pipeline_cv = Pipeline( steps = [
                    ('feature_generator' , FeatureGenerator()),
                    ('CountVectorizer', CountVectorizer(min_df=0,max_df=1,binary=False,ngram_range=(1,2), max_features= 150000))])
            feature_pipeline_tf = Pipeline( steps = [
                    ('feature_generator' , FeatureGenerator()),
                    ('TfidfVectorizer', TfidfVectorizer(stop_words='english', ngram_range=(1, 2), lowercase=True, max_features=150000))])
            
            preprocessing = ColumnTransformer([('feature_generator_cv', feature_pipeline_cv, review_column),
                    ('feature_generator_tv', feature_pipeline_tf, review_column) ])
            return preprocessing
        except Exception as e:
            raise App_Exception(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info(f"Obtaining preprocessing object.")
            preprocessing_obj = self.get_preprocessing_obj()

            logging.info(f"Obtaining training and test file data.")
            train_collection_name = self.data_transformation_config.ingested_train_collection
            test_collection_name = self.data_transformation_config.ingested_test_collection
            train_conn = MongoDB(train_collection_name , drop_collection=False)
            test_conn = MongoDB(test_collection_name , drop_collection=False)


            logging.info(f"Loading training and test data as pandas dataframe.")
            train_df = load_data_from_mongodb(train_conn)

            test_df = load_data_from_mongodb(test_conn)

            target_column_name = "sentiment"

            logging.info(f"Splitting input and target feature from training and testing dataframe.")
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.hstack((input_feature_train_arr, np.array(target_feature_train_df)))

            test_arr = np.hstack((input_feature_test_arr, np.array(target_feature_test_df)))


            transformed_train_file_path = self.data_transformation_config.transformed_train_file_path
            transformed_test_file_path = self.data_transformation_config.transformed_test_file_path


            logging.info(f"Saving transformed training and testing array.")

            save_numpy_array_data(file_path=transformed_train_file_path, array=train_arr)
            save_numpy_array_data(file_path=transformed_test_file_path, array=test_arr)

            preprocessing_obj_file_path = self.data_transformation_config.preprocessed_object_file_path

            logging.info(f"Saving preprocessing object.")
            save_object(file_path=preprocessing_obj_file_path, obj=preprocessing_obj)

            data_transformation_artifact = DataTransformationArtifact(is_transformed=True,
                                                                      message="Data transformation successfully.",
                                                                      transformed_train_file_path=transformed_train_file_path,
                                                                      transformed_test_file_path=transformed_test_file_path,
                                                                      preprocessed_object_file_path=preprocessing_obj_file_path

                                                                      )
            logging.info(f"Data transformations artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise App_Exception(e, sys) from e

    def __del__(self):
        logging.info(f"{'>>' * 30}Data Transformation log completed.{'<<' * 30} \n\n")


if __name__ == "__main__":
    config = Configuration()
    data_transformation_config = config.get_data_transformation_config()
    
    data_transformation_obj = DataTransformation(data_transformation_config)
    
    data_transformation_artifact = data_transformation_obj.initiate_data_transformation()
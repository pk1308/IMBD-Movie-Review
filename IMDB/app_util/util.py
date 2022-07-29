
import yaml
from IMDB.app_exception.exception import App_Exception
from IMDB.app_database.mongoDB import MongoDB
from IMDB.app_logger import App_Logger
import numpy as np
import dill
import pandas as pd
import boto3
import botocore
import os 

S3_BUCKET_NAME = "pk1308mlproject"


logging = App_Logger(__name__)


def write_yaml_file(file_path: str, data: dict = None):
    """
    Create yaml file 
    file_path: str
    data: dict
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as yaml_file:
            if data is not None:
                yaml.dump(data, yaml_file)
    except Exception as e:
        raise App_Exception(e, sys)


def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML file and returns the contents as a dictionary.
    file_path: str
    """
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise App_Exception(e, sys) from e


def save_numpy_array_data(file_path: str, array: np.array, allow_pickle=True):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise App_Exception(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj, allow_pickle=True)
    except Exception as e:
        raise App_Exception(e, sys) from e


def save_object(file_path: str, obj):
    """
    file_path: str
    obj: Any sort of object
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise App_Exception(e, sys) from e


def load_object(file_path: str):
    """
    file_path: str
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise App_Exception(e, sys) from e

def load_data_from_mongodb(connection_0bj , limit=4000):
    """
    connection_0bj: mongodb connection object
    """
    try:
        data = connection_0bj.Find_Many( query={}, limit=limit)
        load_df = pd.DataFrame(data)
        if "_id" in load_df.columns:
            load_df.drop(columns=["_id"], inplace=True)
            
        return load_df 
    except Exception as e:
        raise App_Exception(e, sys) from e

def s3_download_model(path : str , key_name : str):
    try:
        session = boto3.Session(
        aws_access_key_id= os.environ['AWS_ACCESS_KEY'],
        aws_secret_access_key=os.environ['AWS_ACCESS_SECRET']
        )   
        #Creating S3 Resource From the Session.
        s3 = session.resource('s3')
        bucket = s3.Bucket(S3_BUCKET_NAME)
        obj = bucket.objects.filter()
        file_key = [i for i in obj if key_name in i.key][0]
        bucket.download_file(file_key.key, path) # save to same path
        logging.info("Downloaded Model From S3")

    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise App_Exception(e, sys) from e
        
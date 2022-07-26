from datetime import datetime
import os , sys 
import pandas as pd
from IMDB.app_exception.exception import App_Exception
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()


CONNECTION_STRING = os.getenv('MONGODB_CONNSTRING')
DATABASE_NAME = "IMDB"




class App_Logger:
    """Mongo DB logger class"""
    def __init__(self, filename : str , collection_name = 'Log Collection'):
        self.__collection_name = collection_name
        self.__connection_string = CONNECTION_STRING
        self.__database_name = DATABASE_NAME
        self.__collection = self.get_collection()
        self.file = filename

    def get_collection(self):
        """Returns a collection object"""
        client = MongoClient(self.__connection_string)
        db = client[self.__database_name]
        collection = db[self.__collection_name]
        return collection

    def info(self, message ):
        """Logs info message"""
        self.__collection.insert_one({'timestamp': datetime.now() ,
                                      "level" : 'Info', 
                                      "file_name" :self.file ,                           
                                      'message': message, })
    def debug(self, message):
        """Logs debug message"""
        self.__collection.insert_one({'timestamp': datetime.now() ,
                                      "level" : 'Debug', 
                                      "file_name" :self.file ,                           
                                      'message': message, })
    def error(self, message):
        """Logs error message"""
        self.__collection.insert_one({'timestamp': datetime.now() ,
                                      "level" : 'Error', 
                                      "file_name" :self.file ,                           
                                      'message': message, })
    def warning(self, message):
        """log warning"""
        self.__collection.insert_one({'timestamp': datetime.now() ,
                                      'level' : 'Warning',
                                      "file_name" :self.file ,
                                      "message" : message, })
        
        
    def get_data(self):
        """Returns data from the collection"""
        try:
            data = self.collection.find()
            return data
        except Exception as e:
            raise App_Exception(e)

    def get_data_as_df(self):
        """Returns data from the collection as a pandas database"""
        try:
            data = self.collection.find()
            df = pd.DataFrame(list(data))
            return df
        except Exception as e:
            raise App_Exception(e)

    def get_data_as_df_by_date(self, date):
        """Returns data from the collection as a pandas dataframe"""
        try:
            data = self.collection.find({'date': date})
            df = pd.DataFrame(list(data))
            return df
        except Exception as e:
            raise App_Exception(e)
        





# def get_log_dataframe(file_path):
#     data = []
#     with open(file_path) as log_file:
#         for line in log_file.readlines():
#             data.append(line.split("|"))

#     log_df = pd.DataFrame(data)
#     columns = ["Time stamp", "Log Level", "function name", "message"]
#     log_df.columns = columns

#     log_df["log_message"] = log_df['Time stamp'].astype(str) + ":$" + log_df["message"]

#     return log_df[["log_message"]]
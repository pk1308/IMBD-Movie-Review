from collections import namedtuple

DataIngestionConfig = namedtuple("DataIngestionConfig",
                                 ["dataset_download_url" , 
                                  "dataset_download_file_name", "raw_data_file_path",
                                  "raw_file_path_to_ingest" , "ingested_train_file_path", 
                                  "ingested_test_data_path", "ingested_train_collection", 
                                  "ingested_test_collection"])


DataTransformationConfig = namedtuple("DataTransformationConfig", ["transformed_train_file_path",
                                                                   "transformed_test_file_path",
                                                                   "preprocessed_object_file_path" , 
                                                                   "ingested_train_collection" , 
                                                                   'ingested_test_collection'])

ModelTrainerConfig = namedtuple("ModelTrainerConfig",["transformed_train_file_path",
                                                      "transformed_test_file_path",
                                                       "preprocessed_object_file_path",
                                                       "model_config_file_path",
                                                       "base_accuracy","trained_model_file_path"])

ModelEvaluationConfig = namedtuple("ModelEvaluationConfig", ["model_evaluation_collection",
                                                            "trained_model_path",
                                                            "model_evaluated_file_path" ,
                                                            "model_evaluation_dir"])


TrainingPipelineConfig = namedtuple("TrainingPipelineConfig",
                                    ["artifact_dir", "pipeline_name"])
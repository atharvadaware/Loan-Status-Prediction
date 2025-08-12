import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    
    def get_data_transformer_obj(self):

        """
        This function is responsible for data transformation
        """

        
        try:
            numerical_columns=['Dependents', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
            categorical_columns=['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']

            

            num_pipeline=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ("one hot encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")


            preprocessor=ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])


            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("Converted '3+' string to numerical value 3")
            train_df['Dependents'] = train_df['Dependents'].replace('3+',3).astype('float')
            test_df['Dependents'] = test_df['Dependents'].replace('3+',3).astype('float')

            logging.info("Read train and test data successfully")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_obj()

            target_column_name="Loan_Status"
            drop_columns= [target_column_name, 'Loan_ID']

            input_feature_train_df=train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframe")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path= self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e,sys)
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from typing import Union

class datastrategy(ABC):
    @abstractmethod
    def handel_data(self, data:pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        pass

class DataPreProcessingStrategy(datastrategy):
    """
    This is used to preprocess the data
    """
    def handel_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            data = data.drop(
                [
                    "EmployeeCount", 
                    "EmployeeNumber", 
                    "StandardHours", 
                    "Over18",
                ],
            axis=1)

            data["Attrition"] = data["Attrition"].apply(lambda x: 1 if x == "Yes" else 0)
            data['OverTime'] = data['OverTime'].apply(lambda x: 1 if x == "Yes" else 0)

            """
            Extract label columns and performing label encoding on following variables
            """
            # labelencoder = LabelEncoder()
            # columns_to_label_encoded = data[["OverTime"]]
            # label_data = labelencoder.fit_transform(columns_to_label_encoded)
            # label_data = pd.DataFrame(label_data)


            """
            Extract onehot encoded columns and performing OneHotEncoding on following variables            
            """
            onehot = OneHotEncoder()
            columns_to_onehot_encoded = data[['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']]
            enoded_data = onehot.fit_transform(columns_to_onehot_encoded).toarray()
            features_name = onehot.get_feature_names_out(input_features=columns_to_onehot_encoded.columns)
            onehot_data = pd.DataFrame(enoded_data, columns=features_name)
 
            """
            Extract numerical columns 
            """
            numerical_data = data[
                [
                    'Age',
                    'Attrition',
                    'DailyRate',
                    'DistanceFromHome',
                    'Education', 
                    'EnvironmentSatisfaction',
                    'HourlyRate',
                    'JobInvolvement',
                    'JobLevel',
                    'JobSatisfaction',
                    'MonthlyIncome',
                    'MonthlyRate',
                    'NumCompaniesWorked',
                    'OverTime',
                    'PercentSalaryHike',
                    'PerformanceRating',
                    'RelationshipSatisfaction',
                    'StockOptionLevel',
                    'TotalWorkingYears',
                    'TrainingTimesLastYear',
                    'WorkLifeBalance',
                    'YearsAtCompany',
                    'YearsInCurrentRole',
                    'YearsSinceLastPromotion',
                    'YearsWithCurrManager'
                ]
            ]
            
            """
            concatinate labeled_data, onehot data, numericaldata
            """
            data = pd.concat([onehot_data, numerical_data], axis=1)
            # print(data)
            # print(data.shape)
            return data
          
        except Exception as e:
            logging.error(f"Error in preprocessing data {e}")
            raise e
        
class DataDivideStrategy(datastrategy):
    """
    strategy for dividing data into train and test
    """
    def handel_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divide data into train and test
        """
        try:
            X = data.drop(['Attrition'], axis=1)
            y = data['Attrition']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error while dividing data {e}")
            raise e
        
class DataCleaning:
    """
    class for cleaning data and which process the data and divides into train and test
    """
    def __init__(self, data: pd.DataFrame, strategy: datastrategy):
        self.data = data
        self.strategy = strategy

    def handel_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        handel_data
        """
        try:
            return self.strategy.handel_data(self.data)
        except Exception as e:
            logging.error(f"Error in handeling data {e}")
            raise e


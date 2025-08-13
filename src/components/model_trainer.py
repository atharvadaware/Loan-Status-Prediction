import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_obj_path=os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_training(self,train_arr, test_arr):
        try:
            logging.info("Splitting Dependent and Independent variables from train and test data")
            X_train, y_train, X_test, y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models={
                'Logistic Regression' : LogisticRegression(),
                'Support Vector Machine' : SVC(),
                'Random Forest' : RandomForestClassifier(),
                'K-Neighbors' : KNeighborsClassifier(),
                'Decision Tree' : DecisionTreeClassifier()
            }

            '''
            params={
                "Logistic Regression":{
                    'fit_intercept': [True,False],
                    'C': [1,3,5,7,10],
                    'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
                },
                 
                "Support Vector Machine":{
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'gamma': ['scale', 'auto'],
                    'degree' : [3,9,27,5,7]
                },
                

                "Random Forest":{
                    'n_estimators': [100,200,300,400,500],
                    'criterion': ['gini', 'entropy', 'log_loss'],
                    'max_depth': [2,3,4,5,10]
                },

                "K-Neighbors":{
                    'n_neighbors': [5,7,9,10,15],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                    'leaf_size': [30,40,50,60,100]
                },

                "Decision Tree":{
                    'criterion': ['gini', 'entropy', 'log_loss'],
                    'splitter': ['best', 'random'],
                    'max_depth': [4,8,12,16,20]
                }
            }
            '''
            logging.info("Model Training and Evaluation")
            model_report:dict=evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models) #params=params
            print(model_report)
            print('\n=====================================================================================')
            logging.info(f"Model Report : {model_report}")  


            # To get best model score
            best_model_score=max(sorted(model_report.values()))

            #To get best model name
            best_model_name=list(models.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f'Best model found, Model name : {best_model_name}, Accuracy : {best_model_score}')
            print('\n=====================================================================================')
            logging.info(f'Best model found, Model name : {best_model_name}, Accuracy : {best_model_score}')

            save_object(
                file_path=self.model_trainer_config.trained_model_obj_path,
                obj=best_model
            )

            y_pred_train = best_model.predict(X_train)
            y_pred_test = best_model.predict(X_test)

            train_score=accuracy_score(y_train, y_pred_train)
            test_score=accuracy_score(y_test, y_pred_test)

            return ("Training Accuracy", train_score, "Testing Accuracy", test_score)

        except Exception as e:
            raise CustomException(e,sys)
import os
import sys
import dill

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException




def save_object(file_path, obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models): #params):
    try:
        report={}
        for i in range(len(models)):
            model = list(models.values())[i]

            '''
            param_grid=params[list(models.keys())[i]]

            gs = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            '''
            # Train model
            model.fit(X_train,y_train)

            # Predict test data
            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)

            # Get R2 Score for train and test data
            train_model_score=accuracy_score(y_train, y_train_pred)
            test_model_score=accuracy_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score 

        return report
        
    except Exception as e:
        raise CustomException(e,sys)
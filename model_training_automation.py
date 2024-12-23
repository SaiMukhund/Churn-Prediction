import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score,classification_report,ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from preprocessing import data_preprocessing

random_seed=24

def best_model_classification(X_train,y_train,X_test,y_test,save_model_path=None):
    """Find the best model for the data and prints the score of the models and confusion matrix of the best model 

    Args:
        X_train (dataframe): features data that is to be trained 
        y_train (np.array): the acutal target values for train data 
        X_test (dataframe): the features data that is to test model accuracy 
        y_test (np.array): the actual target values for test data 
        save_model_path (_type_, str): to save the best model and path should end with .pkl . Defaults to None.
    """
    models = {
    "Logistic Regression": LogisticRegression(random_state=random_seed),
    "Decision Tree": DecisionTreeClassifier(random_state=random_seed),
    "Random Forest": RandomForestClassifier(random_state=random_seed)
        }

    params = {
        "Logistic Regression": {
            'C': [0.01,0.1, 1, 10,100],
            
        },
        "Decision Tree": {
            'max_depth': list(range(3,30,3)),
            'min_samples_split': list(range(2,11,2)),
            'min_samples_leaf': list(range(1,5,1)),
        },
        "Random Forest": {
            'n_estimators': list(range(50,501,50)),
            'max_depth': list(range(3,30,3)),
            'min_samples_split': list(range(2,11,2)),
            'min_samples_leaf': list(range(1,5,1)),
        }
    }

    # Perform GridSearch for each model
    best_models = {}
    best_scores = {}
    for model_name, model in models.items():
        print(f"Performing GridSearch for {model_name}...")
        grid_search = GridSearchCV(estimator=model, param_grid=params[model_name], cv=4, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_models[model_name] = grid_search.best_estimator_
        best_scores[model_name] = grid_search.best_score_
        print(f"Best CV Accuracy for {model_name}: {grid_search.best_score_:.4f}\n")
        print(f"Best Parameters for {model_name}: {grid_search.best_params_}")
        

    # Select the best model based on CV accuracy
    best_model_name = max(best_scores, key=best_scores.get)
    best_model = best_models[best_model_name]


    print(f"The best model is {best_model_name} with a CV accuracy of {best_scores[best_model_name]:.4f}\n")

    # Save the best model to a file
    if save_model_path!=None:
        joblib.dump(best_model, save_model_path)
        print(f"The best model ({best_model_name}) has been saved as 'best_model.pkl'.")
    
    
    # Evaluate the best model on the test set
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy of the best model ({best_model_name}): {test_accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    ConfusionMatrixDisplay.from_estimator(best_model,X_test,y_test)
    plt.title("Confusion matrix of Random forest model")
    plt.show()
    
    
if __name__=="__main__":
    data=pd.read_csv("./data/final_data.csv")
    preprocessor_model_path="./models/.preprocessor_automated.pkl"
    X_train,X_test,y_train,y_test=data_preprocessing(data,preprocessor_model_path)
    save_model_path="./models/best_model.pkl"
    best_model_classification(X_train,y_train,X_test,y_test,save_model_path)
    
    
    
    
    

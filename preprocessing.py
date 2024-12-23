import pandas as pd
import numpy as np 
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

random_seed=24


def handling_missing_values(data):
    """
    Handles missing values 
    Args:
        data (pandas dataframe): 

    Returns:
        pandas dataframe: 
    """
    
    data=data.copy()
    ## drop if the churn value  Is Null
    data.dropna(subset=["Churn"],inplace=True)
    
    ## drop a row if more than 2 columns are null as it might be difficult to fill the missing values and might deviate from the actual data 
    data.dropna(subset=["Age","Tenure","MonthlyCharges","TotalCharges","PaymentMethod",],thresh=3,inplace=True)
    
    ### getting indices of numerical and categorical in data
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    string_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    ## fill null values of gender with No gender as this has effect on the people with null value as this no gender might have a different behaviour 
    data["Gender"]=data["Gender"].fillna("No Gender")
    ##fill the Payment Method data with mode 
    data["PaymentMethod"].fillna(data["PaymentMethod"].mode(),inplace=True)
    
    ## using knn algorithm to fill the values of Numerical Data 
    imputer = KNNImputer(n_neighbors=5)
    data[numerical_cols] = imputer.fit_transform(data[numerical_cols])
    return data

def data_preprocessing(data,save_preprocessor_path=None):
    """
    Preprocess the data feature Engineering ,scaling ,Encoding 
    Args:
        data (pandas dataframe):
        save_preprocessor_path (string , optional): path to save the preprocsser . Defaults to None.

    Returns:
        pandas dataframe
    """
    
    data=data.copy()
    ## dropping customer as not a useful information 
    data.drop("CustomerID",inplace=True,axis=1)

    ## feature Engineering 
    data["AverageServiceUsage"]=(data[["ServiceUsage1","ServiceUsage2","ServiceUsage3"]]).mean(axis=1)
    data["AverageCharges"]=data["TotalCharges"]/data["Tenure"]
    
    ## Getting independent and dependent Variables 
    X=data.drop("Churn",axis=1)
    y=data["Churn"]
    print(X.head())
    ### getting indices of numerical and categorical in X
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    string_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    #converting y into binary 
    y=np.where(y=="Yes",1,0)
    ## spillting of data 
    X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,test_size=0.3,random_state=random_seed)
    
    ## standard_scaler and onehot encoder
    preprocessor=ColumnTransformer(
        transformers=[
            ("num",StandardScaler(),numerical_cols),
            ("cat",OneHotEncoder(handle_unknown='ignore'),string_cols)
        ]
    )
    preprocessor.fit(X_train)
    features=preprocessor.get_feature_names_out()
    X_train_processed=preprocessor.transform(X_train)
    X_train_processed=pd.DataFrame(X_train_processed,columns=features,index=X_train.index)
    X_test_processed=preprocessor.transform(X_test)
    X_test_processed=pd.DataFrame(X_test_processed,columns=features,index=X_test.index)
    
    ## save the preprocessor 
    if save_preprocessor_path!=None:
        joblib.dump(preprocessor, save_preprocessor_path)
        print(f"sucessfully saved the preprocssor at {save_preprocessor_path}")
    
    return X_train_processed,X_test_processed,y_train,y_test


if __name__=="__main__":
    save_preprocessor_model="./models/xyzkml.pkl" ## give the model path as you wish 
    data=pd.read_csv("./data/final_data.csv",save_preprocessor_model)
    print(data_preprocessing(data))
    
     
    
    
    
    
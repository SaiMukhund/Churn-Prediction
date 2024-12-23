## Data 
- "./data" folder contains the data collected 
- first_data.csv is data which contains 200 samples which were initially given 
- second_data.xlsx  is the data which contains 2000 samples which were given later 
- final_data.csv contains the combined data of the above 2 


## Models
- the models required for api are saved in the "./models" Folder 
  - preprocessor.pkl is the preprocessor used for data preprocessing which will used for preprocessing the data while deployment 
  - randomforest_model.pkl is the model that is traine while manually checking the best parameters 
  - best_model.pkl is the model that is found using Gridsearchcv to best find the model for the data 

## Scripts 
- preprocessing.py contains the script to handle missing values and preprocess the data and save the preprocessor if the model path is given 
- EDA.ipynb contains all the analysis done for the final data 
- model_training_automation.py contains the code for finding best model 
- model_training_manual.ipynb contains detailes graphs and process used for selecting each model hyper paramters 
- recommendation.py contains the script for recommending services to user based on ther service usage , the recommendation used is cosine similarity 
- model_api.py contains the script for deployment of the model 



## Getting models 
- run the preprocessor.py with final_data.csv as data and give the path for saving the preprocessor with .pkl as file 
- run the model_training_automation.py with final_data.csv as data and give the path for saving model  and preprocessor(optional if not done)
- run the model_trianing_manual.ipynb to check the graphs and accuracy for better understanding ( there is a section where the preprocessor and model are saved for deployment)

## Testing the model API 
- install the dependencies from requirements.txt ` pip install -r requirements.txt`
- apis takes the post request whic should contains the features as json/dict and returns a list of predictions 
- run the model_api.py :  `python model_api.py`
- url="http://127.0.0.1:5050"
- params can be of 2 types  but be sure you send correct data types and maintin the consistency 
```
   params_type1={
    "Gender":"Female",
    "Age":34.364871,
    "Tenure":26.832896,
    "MonthlyCharges":87.351601,
    "TotalCharges":9927.133975,
    "PaymentMethod":"PayPal",
    "ServiceUsage1":8.25153,
    "ServiceUsage2":3.75883,
    "ServiceUsage3":88.842903
    }
  params_type2={
    "Gender":["Female",...],
    "Age":[34.364871,...],
    "Tenure":[26.832896,...],
    "MonthlyCharges":[87.351601,....],
    "TotalCharges":[927.133975,....],
    "PaymentMethod":["PayPal",...],
    "ServiceUsage1":[8.25153,....],
    "ServiceUsage2":[3.75883,...],
    "ServiceUsage3":[88.842903,.....]
}
```
- code for getting the predictions 

```
import requests
predictions =requests.post(url=url,json=params_type2).json()["prediction"]
```

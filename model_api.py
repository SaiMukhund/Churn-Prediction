from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Churn Prediction Mode API "

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON
        data = request.get_json()
        
        columns=["Gender","Age","Tenure","MonthlyCharges","TotalCharges","PaymentMethod","ServiceUsage1","ServiceUsage2","ServiceUsage3"]
        # Extract features as a list
        if None in data.values():
            return jsonify({"error": "No features provided. Please include a 'features' key with an array of values."}), 400
        data_dict={}
        instances=[]
        for c in columns:
            data_dict[c]=data.get(c)
            instances.append(isinstance(data_dict[c],list))
        
        bool_and=True 
        bool_or=False
        for i in instances: 
            bool_and=bool_and and i 
            bool_or=bool_or and i
        if bool_and==True:
            data_df=pd.DataFrame(data_dict)
        elif bool_or==False:
            data_df=pd.DataFrame([data_dict])
        else:
            return jsonify({"error": "The values in the Features  have inconsist values please check "}), 400
        
        
        data_df["AverageServiceUsage"]=(data_df[["ServiceUsage1","ServiceUsage2","ServiceUsage3"]]).mean(axis=1)
        data_df["AverageCharges"]=data_df["TotalCharges"]/data["Tenure"]
        
        X=pd.DataFrame(preprocessor.transform(data_df),columns=features)
        

        # Make a prediction
        prediction = model.predict(X)
        
        
        # Return the prediction as a JSON response
        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    ## load the preprocessor used for the training of the model 
    preprocessor_path="./models/preprocessor.pkl" 
    preprocessor=joblib.load(preprocessor_path)
    features=preprocessor.get_feature_names_out()   
    ## load the model
    model_path="./models/randomforest_model.pkl"
    model = joblib.load(model_path)
    
    ## run the api 
    app.run(debug=False,port=5050,host="0.0.0.0")

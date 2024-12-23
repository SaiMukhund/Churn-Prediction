import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


def data_processing_for_services(data):
    """To get the services data out the whole data 

    Args:
        data (pandas dataframe): 

    Returns:
        pandas dataframe : 
    """
    ## getting the  service  usages 
    servicedata=data[["ServiceUsage1","ServiceUsage2","ServiceUsage3"]] 
    ## setting the index and columns names 
    servicedata.index.name="Id"
    servicedata.columns.name="service"
    ## scaling the services for comparison 
    scaler=MinMaxScaler()
    servicedata_norm= pd.DataFrame(scaler.fit_transform(servicedata), columns=servicedata.columns,index=servicedata.index)
    
    return servicedata_norm
    
def Recommender_using_similarity_search(user_id,serviceuser_data,top_n =20,user_similarity_threshold =  0.5,used_service_threshold=0.4):
    """Recommendation based on cosine similarity user based Collaborative filtering 

    Args:
        user_id (int ): the user id for which  recommendation is given 
        serviceuser_data (pandas dataframe): the dataframe which contains the customer id ans service usage 
        top_n (int, optional): number of similar customers . Defaults to 20.
        user_similarity_threshold (float, optional): threshold for finding similar customer above it . Defaults to 0.5.
        used_service_threshold (float, optional): usage of service which is considered to be used highly . Defaults to 0.4.

    Returns:
        list of services recommendation
    """
    ## giving service usage to 0 if nan 
    serviceuser_data.fillna(0,inplace=True)
    
    ##calculating the similarity scores between the users 
    similarity_matrix = cosine_similarity(serviceuser_data)
    similarity_matrix_df = pd.DataFrame(similarity_matrix, index=serviceuser_data.index, columns=serviceuser_data.index)
    ## dropping the user itsef as  we dont need 
    similarities = similarity_matrix_df[user_id].drop(user_id)
    
    ## calculating the similar users to user_id 
    top_similar_users = similarity_matrix_df[similarity_matrix_df[user_id]>user_similarity_threshold][user_id].sort_values(ascending=False)[:top_n]
    similar_user_services = serviceuser_data[serviceuser_data.index.isin(top_similar_users.index)].dropna(axis=1, how='all')
    
    ##getting the services used by the user enoough that is by having a threshold 
    used_service = serviceuser_data.loc[serviceuser_data.index== user_id, serviceuser_data.loc[user_id,:]>used_service_threshold]
    
    service_score = {}
    # loop through each service in the similar user to get 
    for i in similar_user_services.columns:
        service_usage = similar_user_services[i] 
        total = 0  
        count = 0
        for u in top_similar_users.index:
            if u in service_usage.index and not pd.isna(service_usage[u]):
                score = top_similar_users[u] * service_usage[u] 
                total+= score
                count +=1
        service_score[i] = total / count  

    # Convert dictionary to pandas dataframe
    service_score = pd.DataFrame(service_score.items(), columns=['service_name', 'similarity_score'])
    used_service_list = list(used_service.columns)
    service_score = service_score[~service_score['service_name'].isin(used_service_list)]
        
    # Sort services 
    ranked_service_score = service_score.sort_values(by='similarity_score', ascending=False)
    ranked_service_score['similarity_score'] = ranked_service_score['similarity_score']  / ranked_service_score['similarity_score'].abs().max() 
    ranked_service_score.rename(columns = {"similarity_score": "Recommendation of the service "},  
          inplace = True) 
    return ranked_service_score
    


if __name__=="__main__":
    data=pd.read_csv("./data/final_data.csv")
    serviceuser_data=data_processing_for_services(data)
    user_id=13
    print(Recommender_using_similarity_search(user_id,serviceuser_data))
    
    
    
    
    
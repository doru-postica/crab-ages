
# import necessary libraries
import numpy as np
import pandas as pd

import streamlit as st
import joblib

import xgboost as xgb
from sklearn.neighbors import NearestNeighbors

from custom_functions.create_new_features import create_new_features # local files
from custom_functions.create_polynomials import create_polynomials # local files

# functions needed for app to run
def load_data():
    column_names_initial_input = joblib.load('fit_pickles/columns_for_input.pkl')
    neighbors_database = joblib.load('fit_pickles/database_for_kneighbors.pkl')
    data_similarity_thresholds = joblib.load('fit_pickles/thresholds_for_data_similarity.pkl')
    return (column_names_initial_input, neighbors_database, data_similarity_thresholds)

def load_models():
    fit_imputer = joblib.load('fit_pickles/fit_imputer.pkl')
    fit_scaler = joblib.load('fit_pickles/fit_scaler.pkl')
    fit_model = joblib.load('fit_pickles/MAPIE_fit_model.pkl')
    return (fit_imputer, fit_scaler, fit_model)

def get_streamlit_input():
    st.title('Crab Age Predictor')
    st.image('https://www.visitcurrituck.com/wp-content/uploads/2016/03/ghost-crab.png')
    st.header('Enter the characteristics of the crab that you saw:')
    st.text("If you don't have all the measurements, it's fine. Just mark them as 0.")

    # used min values 0, will treat as missing and impute them
    # set max values according to existing dataset distribution
    # set default slider values as median values from dataset

    sex = st.selectbox("Sex ('I' if you do not know)", ('M', 'F', 'I'))
    length = st.slider('Length in cm: ', min_value=0.0, max_value=2.1, value=1.38)
    diameter = st.slider('Diameter in cm: ', min_value=0.0, max_value=1.45, value=1.07)
    height = st.slider('Height in cm: ', min_value=0.0, max_value=0.55, value=0.36)
    weight = st.slider('Weight in g: ', min_value=0.0, max_value=60.0, value=23.81)
    shucked_weight = st.slider('Shucked_weight in g: ', min_value=0.0, max_value=25.0, value=9.97)
    viscera_weight = st.slider('Viscera_weight in g: ', min_value=0.0, max_value=12.0, value=4.98)
    shell_weight = st.slider('Shell weight in g: ', min_value=0.0, max_value=15.0, value=6.94)
    confidence_level = st.slider('(Optional) My desired confidence level is: ', min_value=0.01, max_value=0.99, value=0.9)  
    return (sex, length, diameter, height, weight, shucked_weight, viscera_weight, shell_weight, confidence_level)

def prepare_data(input_array): # takes initial array and returns a dataframe ready for our model       
    mydf = pd.DataFrame(data=[input_array], columns = column_names_initial_input) # convert to DataFrame
    mydf['Sex'] = pd.Categorical(mydf['Sex'], categories=['F', 'I', 'M']) # so that get dummies will create columns of all possible combinations, not just one
    mydf.replace(0, np.nan, inplace=True) # in case the user did not have some inputs, we replace with nan and use KNNImputer on them
    
    create_polynomials(mydf) # same feature engineering as in trained model
    create_new_features(mydf) # same feature engineering as in trained model
    
    mydf_numerical = mydf.drop(['Sex'], axis=1)
    mydf_categorical = mydf[['Sex']]
    
    mydf_numerical = pd.DataFrame(data=fit_scaler.transform(mydf_numerical), #scale
                                  index=mydf_numerical.index,
                                  columns = mydf_numerical.columns)
    mydf_categorical = pd.get_dummies(mydf_categorical)
    
    mydf = pd.concat([mydf_categorical, mydf_numerical], axis=1)
    
    mydf = pd.DataFrame(data=fit_imputer.transform(mydf),
                        columns=mydf.columns,
                        index=mydf.index) # impute missing data   
    return mydf

def assess_similarity(user_input_dataframe): #gets called inside get_output function only
    merged_df = pd.concat([neighbors_database,my_df],ignore_index=True, axis=0) # user input is the last row in merged_df
    neigh = NearestNeighbors(n_neighbors=5, n_jobs=-1)
    neigh.fit(merged_df)
    dist, ind = neigh.kneighbors(merged_df.iloc[[-1]]) # we just need neighbors distance for user input, the last row of df
    return dist.mean()


def get_output(my_df, confidence_level):  
    age, confidence_intervals = fit_model.predict(my_df, alpha=1-confidence_level)
    similarity = assess_similarity(my_df)

    if similarity > data_similarity_thresholds[1]:
        st.success('Whoa! Your measurements are out of this world. Even though the model could predict something, your inputs are highly unlikely to be true!')
    elif similarity > data_similarity_thresholds[0]:
        st.success('Hmm.. Comparing your inputs with our database, it seems this crab is a bit of an outlier. Please take the prediction with a grain of salt.')
        st.success('The predicted age of the crab you saw is {} months.'.format(age[0].astype('int')))
        st.success('We can say that, with {}% confidence, the age is between {} and {} months.'.format(int(confidence_level*100), confidence_intervals[:,0][0][0].astype('int'), confidence_intervals[:,1][0][0].astype('int')))
    else:
        st.success('The predicted age of the crab you saw is {} months.'.format(age[0].astype('int')))
        st.success('We can say that, with {}% confidence, the age is between {} and {} months.'.format(int(confidence_level*100), confidence_intervals[:,0][0][0].astype('int'), confidence_intervals[:,1][0][0].astype('int')))        


# Executing script:

column_names_initial_input, neighbors_database, data_similarity_thresholds = load_data()
fit_imputer, fit_scaler, fit_model = load_models()
sex, length, diameter, height, weight, shucked_weight, viscera_weight, shell_weight, confidence_level = get_streamlit_input()

if st.button('Predict age (in months)'):
    my_df = prepare_data([sex, length, diameter, height, weight, shucked_weight, viscera_weight, shell_weight])   
    get_output(my_df, confidence_level)
    

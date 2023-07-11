
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import xgboost as xgb
from sklearn.neighbors import NearestNeighbors

from custom_functions.create_new_features import create_new_features
from custom_functions.create_polynomials import create_polynomials


def load_data():
    initial_column_names = joblib.load('fit_pickles/columns_for_input.pkl')
    neighbors_database = joblib.load('fit_pickles/database_for_kneighbors.pkl')
    similarity_levels = joblib.load('fit_pickles/thresholds_for_data_similarity.pkl')
    return (initial_column_names, neighbors_database, similarity_levels)


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


def prepare_data(input_array):
    mydf = pd.DataFrame(data=[input_array], columns=initial_column_names)
    mydf['Sex'] = pd.Categorical(mydf['Sex'], categories=['F', 'I', 'M'])
    mydf.replace(0, np.nan, inplace=True)

    create_polynomials(mydf)
    create_new_features(mydf)

    mydf_numerical = mydf.drop(['Sex'], axis=1)
    mydf_categorical = mydf[['Sex']]

    mydf_numerical = pd.DataFrame(data=fit_scaler.transform(mydf_numerical),
                                  index=mydf_numerical.index,
                                  columns=mydf_numerical.columns)
    mydf_categorical = pd.get_dummies(mydf_categorical)

    mydf = pd.concat([mydf_categorical, mydf_numerical], axis=1)

    mydf = pd.DataFrame(data=fit_imputer.transform(mydf),
                        columns=mydf.columns,
                        index=mydf.index)
    return mydf


def assess_similarity(mydf):  # gets called inside get_output function only
    merged_df = pd.concat([neighbors_database, my_df], ignore_index=True, axis=0)  # user input is the last row
    neigh = NearestNeighbors(n_neighbors=5, n_jobs=-1)
    neigh.fit(merged_df)
    dist, ind = neigh.kneighbors(merged_df.iloc[[-1]])
    return dist.mean()


def get_output(my_df, confidence_level):
    age, confidence_intervals = fit_model.predict(my_df, alpha=1-confidence_level)
    similarity = assess_similarity(my_df)

    if similarity > similarity_levels[1]:
        st.success('Whoa! Your measurements are out of this world. Even though the model could predict something, your inputs are highly unlikely to be true!')
    elif similarity > similarity_levels[0]:
        st.success('Hmm.. Comparing your inputs with our database, it seems this crab is a bit of an outlier. Please take the prediction with a grain of salt.')
        st.success('The predicted age of the crab you saw is {} months.'.format(age[0].astype('int')))
        st.success('We can say that, with {}% confidence, the age is between {} and {} months.'.format(int(confidence_level*100), confidence_intervals[:, 0][0][0].astype('int'), confidence_intervals[:, 1][0][0].astype('int')))
    else:
        st.success('The predicted age of the crab you saw is {} months.'.format(age[0].astype('int')))
        st.success('We can say that, with {}% confidence, the age is between {} and {} months.'.format(int(confidence_level*100), confidence_intervals[:, 0][0][0].astype('int'), confidence_intervals[:, 1][0][0].astype('int')))


# Executing script:

initial_column_names, neighbors_database, similarity_levels = load_data()
fit_imputer, fit_scaler, fit_model = load_models()
sex, length, diameter, height, weight, shucked_weight, viscera_weight, shell_weight, confidence_level = get_streamlit_input()

if st.button('Predict age (in months)'):
    my_df = prepare_data([sex, length, diameter, height, weight, shucked_weight, viscera_weight, shell_weight])
    get_output(my_df, confidence_level)

import streamlit as st 
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport
from pycaret.classification import setup, pull, compare_models, save_model, load_model

import pandas as pd
import os

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)


with st.sidebar:
    st.title("AutoML Data Science")
    radio_choice = st.radio(
        "Data Science Flow Process", 
        ["Data Preparation", "EDA", "Modelling", "Prediction", "Download Model"]
        )
    
if radio_choice == "Data Preparation":
    st.title("Data Preparation")
    
    file = st.file_uploader("Upload Dataset")

    if file:
        try:
            df = pd.read_csv(file, index_col=None)
            df.to_csv('dataset.csv', index=None)
            st.dataframe(df)

        except:
            st.warning("Fail to upload dataset.")
            st.warning("Please upload csv file only!")


    
if radio_choice == "EDA":
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

if radio_choice == "Modelling":
    st.title("Machine Learning Model")
    target_column = st.selectbox('Choose the Target Column', df.columns)

    if st.button('Run Modelling'):
        setup(df, target=target_column)
        setup_df = pull()
        st.dataframe(setup_df)
        
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model,'best_model')

if radio_choice == "Prediction":
    st.title("Prediction Using Trained Model")

    loaded_model = load_model('best_model.pkl')
    df_predict = df.copy()
    df_predict.drop(columns="Survived",  axis=1, inplace=True)
    st.dataframe(df_predict)
    predictions = loaded_model.predict(df_predict)
    st.info(predictions)

    df_predict['Prediction'] = predictions
    st.dataframe(df_predict)



    
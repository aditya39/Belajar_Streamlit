import streamlit as st
import pandas as pd
import pycaret.classification as pc
import matplotlib.pyplot as plt
import seaborn as sns
import time
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport

# Title and description
st.title('Data Science Project App')
st.write('This app allows you to load data, preprocess/clean it, train a machine learning model, visualize data, and make inferences with new data.')

# Sidebar for navigation
st.sidebar.title("Navigation")
sections = ["Load Data", 
            "Data Analyze", 
            "Data Preprocessing/Cleaning", 
            "Machine Learning Training", 
            "Visualize Data", 
            "Inference"]
section = st.sidebar.radio("Go to", sections)

# Load Data
if section == "Load Data":
    st.header("Load Data")
    data_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if st.button('Load Data'): # Kalo button diklik...
        if data_file is not None:
            data = pd.read_csv(data_file)
            st.session_state['data'] = data
            st.write("Data Loaded Successfully")
            st.write(data.head())
            # st.dataframe(data)
        else:
            st.write("Please upload a CSV file.")

# Data Analyze (pandas profilling)
elif section == "Data Analyze":
    if 'data' in st.session_state:
        data = st.session_state['data']
        profile_df = data.profile_report()
        st_profile_report(profile_df)


# Data Preprocessing/Cleaning
elif section == "Data Preprocessing/Cleaning":
    st.header("Data Preprocessing/Cleaning")
    if 'data' in st.session_state:
        st.write("Original Data")
        st.write(st.session_state['data'].head())
        
        # Preprocessing options
        st.write("Select preprocessing steps:")
        drop_missing = st.checkbox("Drop missing values")
        fill_missing = st.checkbox("Fill missing values with mean")
        normalize_data = st.checkbox("Normalize data")
        drop_columns = st.multiselect("Select columns to drop", st.session_state['data'].columns)
        
        if st.button('Preprocess Data'):
            data = st.session_state['data'].copy()
            
            if drop_missing:
                data = data.dropna()
                st.write("Dropped missing values")
            
            if fill_missing:
                data = data.fillna(data.mean())
                st.write("Filled missing values with mean")
            
            if normalize_data:
                data = (data - data.mean()) / data.std()
                st.write("Normalized data")
                
            if drop_columns:
                data = data.drop(columns=drop_columns)
                st.write(f"Dropped columns: {drop_columns}")
            
            st.session_state['processed_data'] = data
            st.write("Processed Data")
            st.write(data.head())
            
            # Visualization: Missing values heatmap
            if drop_missing or fill_missing:
                st.write("Missing Values Heatmap")
                plt.figure(figsize=(10, 4))
                sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
                st.pyplot(plt)
            
            # Visualization: Data distribution
            st.write("Data Distribution after Cleaning")
            plt.figure(figsize=(10, 4))
            data.hist(bins=30, figsize=(10, 10))
            st.pyplot(plt)
    else:
        st.write("Please load the data first")

# Machine Learning Training
elif section == "Machine Learning Training":
    st.header("Machine Learning Training")
    if 'processed_data' in st.session_state:
        if st.button('Train Model'):
            data = st.session_state['processed_data']
            st.write("Setting up PyCaret...")
            
            with st.spinner('Setting up PyCaret...'):
                # Setup the PyCaret environment
                setup = pc.setup(data, target='Survived', fold=5)  # Adjust 'target' as needed
                
            st.write("Comparing models...")
            progress_bar = st.progress(0)
            progress_steps = 100 / 5
            
            # Compare models and get their performance
            for i in range(1, 101):
                time.sleep(0.1)  # Simulating the training process
                progress_bar.progress(i)
            
            best_model = pc.compare_models()
            model_comparison = pc.pull()
            
            st.session_state['best_model'] = best_model
            st.session_state['model_comparison'] = model_comparison
            st.write("Best Model Trained:", best_model)
            
            # Display model comparison
            st.write("Model Comparison:")
            st.write(model_comparison)
            
            # Display PyCaret plots
            st.write("Model Performance Plots")
            plot_types = ["auc", "confusion_matrix", "feature", "learning", "manifold"]
            selected_plot = st.selectbox("Select plot type", plot_types)
            
            if st.button('Show Plot'):
                if selected_plot:
                    plot = pc.plot_model(best_model, plot=selected_plot, save=True)
                    st.image(f'./{selected_plot}.png', caption=f'{selected_plot.capitalize()} Plot')
    else:
        st.write("Please preprocess the data first")

# Visualize Data
elif section == "Visualize Data":
    st.header("Visualize Data")
    if 'data' in st.session_state:
        st.write("Data Visualization")
        visualization_type = st.selectbox("Choose a visualization", ["Histogram", "Scatter Plot", "Correlation Heatmap"])
        
        if st.button('Show Visualization'):
            data = st.session_state['data']
            if visualization_type == "Histogram":
                column = st.selectbox("Select Column", data.columns)
                plt.figure(figsize=(10, 4))
                sns.histplot(data[column])
                st.pyplot(plt)
            elif visualization_type == "Scatter Plot":
                col1 = st.selectbox("Select X Column", data.columns)
                col2 = st.selectbox("Select Y Column", data.columns)
                plt.figure(figsize=(10, 4))
                sns.scatterplot(x=data[col1], y=data[col2])
                st.pyplot(plt)
            elif visualization_type == "Correlation Heatmap":
                plt.figure(figsize=(10, 4))
                sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
                st.pyplot(plt)
    else:
        st.write("Please load the data first")

# Inference
elif section == "Inference":
    st.header("Inference with New Data")
    if 'best_model' in st.session_state:
        input_data = {}
        
        new_data_file = st.file_uploader("Upload new data for inference", type=["csv"])
        if new_data_file is not None:
            new_data = pd.read_csv(new_data_file)

            drop_columns = st.multiselect("Select columns to drop", st.session_state['data'].columns)
            if drop_columns:
                dropped_data = new_data.drop(columns=drop_columns)
                st.write(f"Dropped columns: {drop_columns}")
            
            if st.button('Predict Test Data'):
                model = st.session_state['best_model']
                predictions = pc.predict_model(model, data=dropped_data)
                st.write("Predictions:")
                st.write(predictions)

        
        if 'data' in st.session_state:
            columns = st.session_state['data'].columns
            for col in columns:
                if col != 'target':  # Adjust as needed
                    input_data[col] = st.text_input(f"Input for {col}")
        
        if st.button('Predict'):
            model = st.session_state['best_model']
            input_df = pd.DataFrame([input_data])
            predictions = pc.predict_model(model, data=input_df)
            st.write("Predictions:")
            st.write(predictions)
    else:
        st.write("Please train the model first")


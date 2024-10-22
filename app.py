import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load CSV data
def load_data(file):
    data = pd.read_csv(file)
    return data

# Function to display basic statistics
def display_statistics(df):
    st.write("Descriptive Statistics:")
    st.write(df.describe())

# Function to visualize data
def visualize_data(df):
    st.write("Data Visualization:")
    
    # Heatmap of correlations
    corr = df.corr()
    st.write("Correlation Heatmap:")
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    st.pyplot()

# Function to split data for training and testing
def split_data(df, target_column):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

# Function for classification using different models
def classification_analysis(X_train, X_test, y_train, y_test):
    # Random Forest Classifier
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    st.write("Random Forest Classifier Report:")
    st.write(classification_report(y_test, rf_pred))

    # SVM Classifier
    svm = SVC()
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)
    st.write("SVM Classifier Report:")
    st.write(classification_report(y_test, svm_pred))

    # XGBoost Classifier
    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    st.write("XGBoost Classifier Report:")
    st.write(classification_report(y_test, xgb_pred))

# Function for regression analysis
def regression_analysis(X_train, X_test, y_train, y_test):
    # Random Forest Regressor
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    st.write("Random Forest Regressor MAE:", mean_absolute_error(y_test, rf_pred))
    st.write("Random Forest Regressor RMSE:", mean_squared_error(y_test, rf_pred, squared=False))

    # SVM Regressor
    svr = SVR()
    svr.fit(X_train, y_train)
    svr_pred = svr.predict(X_test)
    st.write("SVM Regressor MAE:", mean_absolute_error(y_test, svr_pred))
    st.write("SVM Regressor RMSE:", mean_squared_error(y_test, svr_pred, squared=False))

    # XGBoost Regressor
    xgb = XGBRegressor()
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    st.write("XGBoost Regressor MAE:", mean_absolute_error(y_test, xgb_pred))
    st.write("XGBoost Regressor RMSE:", mean_squared_error(y_test, xgb_pred, squared=False))

# Streamlit app
st.title('CSV Data Visualization, Analysis, and Forecasting with Machine Learning')

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    st.write("Data Preview:")
    st.dataframe(df)
    
    if st.checkbox("Show Descriptive Statistics"):
        display_statistics(df)
    
    if st.checkbox("Visualize Data"):
        visualize_data(df)

    # Select target column for prediction
    target_column = st.selectbox("Select Target Column for Prediction", df.columns)

    if st.checkbox("Perform Classification Analysis"):
        X_train, X_test, y_train, y_test = split_data(df, target_column)
        classification_analysis(X_train, X_test, y_train, y_test)

    if st.checkbox("Perform Regression Analysis"):
        X_train, X_test, y_train, y_test = split_data(df, target_column)
        regression_analysis(X_train, X_test, y_train, y_test)

    if st.checkbox("Show Feature Importance (Random Forest)"):
        rf = RandomForestRegressor()
        X_train, X_test, y_train, y_test = split_data(df, target_column)
        rf.fit(X_train, y_train)
        importances = rf.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})
        st.write(feature_importance_df.sort_values('Importance', ascending=False))

# Future scope: You can add a forecasting feature here, for example using ARIMA or other models based on time series data.

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os

# Set page configuration
st.set_page_config(page_title="Employee Salary Prediction", layout="wide")

# Initialize session state for prediction history
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

# Title
st.title("Employee Salary Prediction App")

# Load dataset
@st.cache_data
def load_data(file_path="Employee_Dataset.csv"):
    try:
        df = pd.read_csv(file_path)
        expected_columns = ['Age', 'Experience', 'Education', 'Department', 'Location', 'Company Type', 'Performance Score', 'Last Hike %', 'Salary']
        if not all(col in df.columns for col in expected_columns):
            st.error(f"Dataset must contain columns: {expected_columns}")
            return None
        return df
    except FileNotFoundError:
        st.error(f"{file_path} not found. Please ensure the file is in the same directory or upload a CSV.")
        return None

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    expected_columns = ['Age', 'Experience', 'Education', 'Department', 'Location', 'Company Type', 'Performance Score', 'Last Hike %', 'Salary']
    if not all(col in df.columns for col in expected_columns):
        st.error(f"Uploaded dataset must contain columns: {expected_columns}")
        st.stop()
else:
    df = load_data()
    if df is None:
        st.stop()

# Load models and preprocessor
try:
    preprocessor = joblib.load("preprocessor.pkl")
    lr_model = joblib.load("linear_regression_model.pkl")
    rf_model = joblib.load("random_forest_model.pkl")
    dl_model = load_model("deep_learning_model.keras")
except FileNotFoundError:
    st.warning("Model or preprocessor files not found. Please retrain models using the button below.")

# Retrain models
st.sidebar.header("Retrain Models")
if st.sidebar.button("Retrain Models"):
    with st.spinner("Retraining models..."):
        selected_columns = ['Age', 'Experience', 'Education', 'Department', 'Location', 'Company Type', 'Performance Score', 'Last Hike %', 'Salary']
        df = df[selected_columns]
        numeric_features = ['Age', 'Experience', 'Performance Score', 'Last Hike %']
        categorical_features = ['Education', 'Department', 'Location', 'Company Type']

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False))
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        X = df.drop('Salary', axis=1)
        y = df['Salary']
        X_preprocessed = preprocessor.fit_transform(X)
        joblib.dump(preprocessor, 'preprocessor.pkl')

        X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

        # Train Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)

        # Train Random Forest
        rf = RandomForestRegressor(random_state=42)
        param_grid = {'n_estimators': [100], 'max_depth': [10, None], 'min_samples_split': [2]}
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        rf_model = grid_search.best_estimator_

        # Train Deep Learning
        dl_model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(1)
        ])
        dl_model.compile(optimizer='adam', loss='mse')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        dl_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)

        # Save models
        joblib.dump(lr_model, 'linear_regression_model.pkl')
        joblib.dump(rf_model, 'random_forest_model.pkl')
        dl_model.save('deep_learning_model.keras')
        st.success("Models retrained and saved successfully!")

# Compute dynamic model performance
@st.cache_resource
def compute_model_performance(df, _preprocessor, _lr_model, _rf_model, _dl_model):
    try:
        X = df.drop('Salary', axis=1)
        y = df['Salary']
        X_transformed = _preprocessor.transform(X)
        lr_preds = _lr_model.predict(X_transformed)
        rf_preds = _rf_model.predict(X_transformed)
        dl_preds = _dl_model.predict(X_transformed, verbose=0).flatten()
        r2_scores = {
            "Linear Regression": r2_score(y, lr_preds),
            "Random Forest": r2_score(y, rf_preds),
            "Deep Learning": r2_score(y, dl_preds)
        }
        mse_scores = {
            "Linear Regression": mean_squared_error(y, lr_preds),
            "Random Forest": mean_squared_error(y, rf_preds),
            "Deep Learning": mean_squared_error(y, dl_preds)
        }
        return r2_scores, mse_scores
    except:
        return None, None

r2_scores, mse_scores = compute_model_performance(df, preprocessor, lr_model, rf_model, dl_model)
if r2_scores is None:
    st.warning("Cannot compute model performance. Please retrain models.")
    r2_scores = {"Linear Regression": 0, "Random Forest": 0, "Deep Learning": 0}
    mse_scores = {"Linear Regression": 0, "Random Forest": 0, "Deep Learning": 0}

# Sidebar for user inputs
st.sidebar.header("Enter Employee Details")
age = st.sidebar.number_input("Age", min_value=18, max_value=65, value=30, step=1)
experience = st.sidebar.number_input("Experience (years)", min_value=0, max_value=40, value=5, step=1)
education = st.sidebar.selectbox("Education", ["High School", "Bachelor’s", "Master’s", "PhD"])
department = st.sidebar.selectbox("Department", ["HR", "IT", "Sales"])
location = st.sidebar.selectbox("Location", ["New York", "Los Angeles", "Chicago"])
company_type = st.sidebar.selectbox("Company Type", ["Startup", "MNC"])
performance_score = st.sidebar.number_input("Performance Score", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
last_hike = st.sidebar.number_input("Last Hike %", min_value=0.0, max_value=20.0, value=5.0, step=0.1)

# Input validation
if experience > age - 16:
    st.sidebar.warning("Experience seems too high for the given age (assuming work starts at age 16). Please adjust.")

# Create input dataframe
input_data = pd.DataFrame({
    "Age": [age],
    "Experience": [experience],
    "Education": [education],
    "Department": [department],
    "Location": [location],
    "Company Type": [company_type],
    "Performance Score": [performance_score],
    "Last Hike %": [last_hike]
})

# Transform input data
try:
    input_transformed = preprocessor.transform(input_data)
except ValueError as e:
    st.error(f"Error in preprocessing: {e}")
    st.stop()

# Make predictions
try:
    lr_pred = lr_model.predict(input_transformed)[0]
    rf_pred = rf_model.predict(input_transformed)[0]
    dl_pred = dl_model.predict(input_transformed, verbose=0)[0][0]
except:
    st.error("Error making predictions. Please retrain models.")
    lr_pred, rf_pred, dl_pred = 0, 0, 0

# Store predictions in session state
if st.sidebar.button("Predict"):
    st.session_state.predictions.append({
        "Age": age,
        "Experience": experience,
        "Education": education,
        "Department": department,
        "Location": location,
        "Company Type": company_type,
        "Performance Score": performance_score,
        "Last Hike %": last_hike,
        "Linear Regression": lr_pred,
        "Random Forest": rf_pred,
        "Deep Learning": dl_pred
    })

# Reset prediction history
if st.sidebar.button("Clear Prediction History"):
    st.session_state.predictions = []
    st.success("Prediction history cleared.")

# Model selection
model_choice = st.selectbox("Select Model for Prediction", ["All Models", "Linear Regression", "Random Forest", "Deep Learning"])

# Display predictions
st.header("Salary Predictions")
if model_choice == "All Models":
    col1, col2, col3 = st.columns(3)
    col1.metric("Linear Regression", f"${lr_pred:,.2f}")
    col2.metric("Random Forest", f"${rf_pred:,.2f}")
    col3.metric("Deep Learning", f"${dl_pred:,.2f}")
else:
    if model_choice == "Linear Regression":
        st.metric(model_choice, f"${lr_pred:,.2f}")
    elif model_choice == "Random Forest":
        st.metric(model_choice, f"${rf_pred:,.2f}")
    else:
        st.metric(model_choice, f"${dl_pred:,.2f}")

# Download predictions
if st.session_state.predictions:
    predictions_df = pd.DataFrame(st.session_state.predictions)
    csv = predictions_df.to_csv(index=False)
    st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")

# Model performance
st.header("Model Performance")
col1, col2 = st.columns(2)

# R² Score Plot
fig_r2, ax_r2 = plt.subplots()
ax_r2.bar(r2_scores.keys(), r2_scores.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax_r2.set_title("Model R² Scores")
ax_r2.set_ylabel("R² Score")
ax_r2.set_ylim(0, 1)
col1.pyplot(fig_r2)

# MSE Plot
fig_mse, ax_mse = plt.subplots()
ax_mse.bar(mse_scores.keys(), mse_scores.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax_mse.set_title("Model MSE")
ax_mse.set_ylabel("Mean Squared Error")
col2.pyplot(fig_mse)

# Random Forest Feature Importance
st.header("Random Forest Feature Importance")
try:
    feature_names = preprocessor.get_feature_names_out()
    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": rf_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    fig_imp, ax_imp = plt.subplots()
    ax_imp.barh(feature_importance["Feature"], feature_importance["Importance"], color='#ff7f0e')
    ax_imp.set_title("Feature Importance (Random Forest)")
    ax_imp.set_xlabel("Importance")
    st.pyplot(fig_imp)
except:
    st.warning("Cannot display feature importance. Please retrain models.")

# Actual vs. Predicted Salary Plot (Random Forest)
st.header("Actual vs. Predicted Salaries (Random Forest)")
try:
    X = df.drop('Salary', axis=1)
    y = df['Salary']
    X_transformed = preprocessor.transform(X)
    rf_preds = rf_model.predict(X_transformed)
    fig_scatter, ax_scatter = plt.subplots()
    ax_scatter.scatter(y, rf_preds, alpha=0.5, color='#ff7f0e')
    ax_scatter.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    ax_scatter.set_xlabel("Actual Salary")
    ax_scatter.set_ylabel("Predicted Salary")
    ax_scatter.set_title("Actual vs. Predicted Salary (Random Forest)")
    st.pyplot(fig_scatter)
except:
    st.warning("Cannot display actual vs. predicted plot. Please retrain models.")

# Feature Distribution Plots
st.header("Feature Distributions")
col1, col2, col3 = st.columns(3)
fig_salary, ax_salary = plt.subplots()
df['Salary'].hist(ax=ax_salary, bins=20, color='#1f77b4')
ax_salary.set_title("Salary Distribution")
ax_salary.set_xlabel("Salary")
ax_salary.set_ylabel("Frequency")
col1.pyplot(fig_salary)

fig_age, ax_age = plt.subplots()
df['Age'].hist(ax=ax_age, bins=20, color='#ff7f0e')
ax_age.set_title("Age Distribution")
ax_age.set_xlabel("Age")
ax_age.set_ylabel("Frequency")
col2.pyplot(fig_age)

fig_exp, ax_exp = plt.subplots()
df['Experience'].hist(ax=ax_exp, bins=20, color='#2ca02c')
ax_exp.set_title("Experience Distribution")
ax_exp.set_xlabel("Experience (years)")
ax_exp.set_ylabel("Frequency")
col3.pyplot(fig_exp)

# Prediction History
if st.session_state.predictions:
    st.header("Prediction History")
    st.dataframe(pd.DataFrame(st.session_state.predictions))

# Option to display dataset
if st.checkbox("Show Dataset"):
    st.header("Dataset")
    st.dataframe(df)
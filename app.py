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

# Set page configuration
st.set_page_config(page_title="Employee Salary Prediction", layout="wide")

# Constants
REQUIRED_COLUMNS = ['Age', 'Experience', 'Education', 'Department', 'Location',
                    'Company Type', 'Performance Score', 'Last Hike %', 'Salary']

# Initialize session state for prediction history
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

st.title("Employee Salary Prediction App")

# Load dataset
@st.cache_data
def load_data(file_path="Employee_Dataset.csv"):
    try:
        df = pd.read_csv(file_path)
        if not all(col in df.columns for col in REQUIRED_COLUMNS):
            st.error(f"Dataset must contain columns: {REQUIRED_COLUMNS}")
            return None
        return df
    except FileNotFoundError:
        st.error(f"{file_path} not found.")
        return None

# Load models
@st.cache_resource
def load_models():
    try:
        preprocessor = joblib.load("preprocessor.pkl")
        lr_model = joblib.load("linear_regression_model.pkl")
        rf_model = joblib.load("random_forest_model.pkl")
        dl_model = load_model("deep_learning_model.keras")
        return preprocessor, lr_model, rf_model, dl_model
    except Exception as e:
        st.warning(f"Error loading models: {e}")
        return None, None, None, None

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
df = pd.read_csv(uploaded_file) if uploaded_file else load_data()
if df is None or not all(col in df.columns for col in REQUIRED_COLUMNS):
    st.error("Invalid or missing dataset.")
    st.stop()

# Load or retrain models
preprocessor, lr_model, rf_model, dl_model = load_models()

st.sidebar.header("Retrain Models")
if st.sidebar.button("Retrain Models"):
    with st.spinner("Retraining models..."):
        df = df[REQUIRED_COLUMNS]
        numeric_features = ['Age', 'Experience', 'Performance Score', 'Last Hike %']
        categorical_features = ['Education', 'Department', 'Location', 'Company Type']

        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False))
        ])
        preprocessor = ColumnTransformer([
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

        X = df.drop('Salary', axis=1)
        y = df['Salary']
        X_preprocessed = preprocessor.fit_transform(X)
        joblib.dump(preprocessor, 'preprocessor.pkl')

        X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

        lr_model = LinearRegression().fit(X_train, y_train)
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(rf, {
            'n_estimators': [100],
            'max_depth': [10, None],
            'min_samples_split': [2]
        }, cv=3, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        rf_model = grid_search.best_estimator_

        dl_model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            BatchNormalization(), Dropout(0.3),
            Dense(32, activation='relu'),
            BatchNormalization(), Dropout(0.2),
            Dense(1)
        ])
        dl_model.compile(optimizer='adam', loss='mse')
        dl_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2,
                     callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)], verbose=0)

        joblib.dump(lr_model, 'linear_regression_model.pkl')
        joblib.dump(rf_model, 'random_forest_model.pkl')
        dl_model.save('deep_learning_model.keras')
        st.success("Models retrained successfully!")

# Compute performance
@st.cache_resource
def compute_model_performance(df, _preprocessor, _lr_model, _rf_model, _dl_model):
    try:
        X = df.drop('Salary', axis=1)
        y = df['Salary']
        X_transformed = _preprocessor.transform(X)
        return {
            "Linear Regression": r2_score(y, _lr_model.predict(X_transformed)),
            "Random Forest": r2_score(y, _rf_model.predict(X_transformed)),
            "Deep Learning": r2_score(y, _dl_model.predict(X_transformed, verbose=0).flatten())
        }, {
            "Linear Regression": mean_squared_error(y, _lr_model.predict(X_transformed)),
            "Random Forest": mean_squared_error(y, _rf_model.predict(X_transformed)),
            "Deep Learning": mean_squared_error(y, _dl_model.predict(X_transformed, verbose=0).flatten())
        }
    except Exception as e:
        st.warning(f"Error computing performance: {e}")
        return None, None

if all([preprocessor, lr_model, rf_model, dl_model]):
    r2_scores, mse_scores = compute_model_performance(df, preprocessor, lr_model, rf_model, dl_model)
else:
    r2_scores = mse_scores = {k: 0 for k in ["Linear Regression", "Random Forest", "Deep Learning"]}

# User inputs
st.sidebar.header("Enter Employee Details")
age = st.sidebar.number_input("Age", 18, 65, 30)
experience = st.sidebar.number_input("Experience (years)", 0, 40, 5)
education = st.sidebar.selectbox("Education", ["High School", "Bachelor’s", "Master’s", "PhD"])
department = st.sidebar.selectbox("Department", ["HR", "IT", "Sales"])
location = st.sidebar.selectbox("Location", ["New York", "Los Angeles", "Chicago"])
company_type = st.sidebar.selectbox("Company Type", ["Startup", "MNC"])
performance_score = st.sidebar.number_input("Performance Score", 1.0, 5.0, 3.0, 0.1)
last_hike = st.sidebar.number_input("Last Hike %", 0.0, 20.0, 5.0, 0.1)

invalid_input = experience > age - 16
if invalid_input:
    st.sidebar.warning("Experience too high for age.")

input_data = pd.DataFrame([{
    "Age": age,
    "Experience": experience,
    "Education": education,
    "Department": department,
    "Location": location,
    "Company Type": company_type,
    "Performance Score": performance_score,
    "Last Hike %": last_hike
}])

# Prediction
if not invalid_input and st.sidebar.button("Predict"):
    try:
        transformed = preprocessor.transform(input_data)
        lr_pred = lr_model.predict(transformed)[0]
        rf_pred = rf_model.predict(transformed)[0]
        dl_pred = dl_model.predict(transformed, verbose=0)[0][0]
        st.session_state.predictions.append({**input_data.iloc[0].to_dict(),
                                             "Linear Regression": lr_pred,
                                             "Random Forest": rf_pred,
                                             "Deep Learning": dl_pred})
    except Exception as e:
        st.error(f"Prediction error: {e}")

# Display predictions
if st.session_state.predictions:
    latest = st.session_state.predictions[-1]
    st.header("Salary Predictions")
    model_choice = st.selectbox("Select Model", ["All Models", "Linear Regression", "Random Forest", "Deep Learning"])
    if model_choice == "All Models":
        col1, col2, col3 = st.columns(3)
        col1.metric("Linear Regression", f"${latest['Linear Regression']:,.2f}")
        col2.metric("Random Forest", f"${latest['Random Forest']:,.2f}")
        col3.metric("Deep Learning", f"${latest['Deep Learning']:,.2f}")
    else:
        st.metric(model_choice, f"${latest[model_choice]:,.2f}")

    predictions_df = pd.DataFrame(st.session_state.predictions)
    st.download_button("Download Predictions", predictions_df.to_csv(index=False), "predictions.csv")

    st.header("Model Performance")
    col1, col2 = st.columns(2)
    fig_r2, ax_r2 = plt.subplots()
    ax_r2.bar(r2_scores.keys(), r2_scores.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax_r2.set_title("Model R² Scores"); ax_r2.set_ylabel("R² Score"); ax_r2.set_ylim(0, 1)
    col1.pyplot(fig_r2)

    fig_mse, ax_mse = plt.subplots()
    ax_mse.bar(mse_scores.keys(), mse_scores.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax_mse.set_title("Model MSE"); ax_mse.set_ylabel("Mean Squared Error")
    col2.pyplot(fig_mse)

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
        st.warning("Cannot display feature importance.")

    try:
        X = df.drop('Salary', axis=1)
        y = df['Salary']
        preds = rf_model.predict(preprocessor.transform(X))
        fig_scatter, ax_scatter = plt.subplots()
        ax_scatter.scatter(y, preds, alpha=0.5, color='#ff7f0e')
        ax_scatter.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        ax_scatter.set_xlabel("Actual Salary")
        ax_scatter.set_ylabel("Predicted Salary")
        ax_scatter.set_title("Actual vs. Predicted Salary (Random Forest)")
        st.pyplot(fig_scatter)
    except:
        st.warning("Cannot plot actual vs. predicted.")

# Optional
if st.checkbox("Show Dataset"):
    st.header("Dataset")
    st.dataframe(df)
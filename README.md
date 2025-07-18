# Employee Salary Prediction App

This project is a **Streamlit web application** for predicting employee salaries using **Linear Regression**, **Random Forest**, and **Deep Learning** models. It processes a dataset (`Employee_Dataset.csv`) with employee features (e.g., Age, Experience, Education) to train models and make predictions. The app supports dataset uploads, model retraining, visualizations (R², MSE, feature importance, scatter plots, histograms), and prediction history export.

It is optimized for datasets of **\~1000–5000 records** and deployed on **Streamlit Cloud**.

---

## 🚀 Features

- **Data Input:** Load default `Employee_Dataset.csv` or upload a custom CSV with required columns.
- **Model Training:** Train Linear Regression, Random Forest, and Deep Learning models.
- **Predictions:** Input employee details to predict salaries using one or all models.
- **Visualizations:** Display R² scores, MSE, Random Forest feature importance, actual vs. predicted salaries, and feature distributions.
- **History:** Store and export prediction history as CSV.
- **Large Dataset Support:** Downsampling and sparse matrix processing for up to 5000 records.

---

## 🛠️ Prerequisites

- **Python:** Version **3.10 recommended** (Python 3.13 may have issues with TensorFlow 2.16.2).
- **Dependencies** (listed in `requirements.txt`):
  ```txt
  streamlit
  pandas
  numpy
  scikit-learn
  joblib
  tensorflow==2.16.2
  matplotlib
  ```
- **Git:** For version control and deployment.
- **Streamlit Cloud Account:** For online deployment.

---

## 📁 Project Structure

```
/salary-prediction-ml
├── app.py                        # Main Streamlit app
├── requirements.txt              # Python dependencies
├── Employee_Dataset.csv          # Default dataset (~1000–5000 records)
├── preprocessor.pkl              # Preprocessor for data transformation
├── linear_regression_model.pkl   # Linear Regression model
├── random_forest_model.pkl       # Random Forest model
├── deep_learning_model.keras     # Deep Learning model
├── y_scaler.pkl                  # Target scaler for Deep Learning
├── generate_dataset.py           # (Optional) Script to generate dataset
└── README.md                     # Project readme file
```

---

## 💻 Setup Instructions

### 🔹 Local Setup

1. **Clone the Repository**

   ```bash
   git clone <your-repo-url>
   cd salary-prediction-ml
   ```

2. **Set Up Virtual Environment**

   ```bash
   python3.10 -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   > If TensorFlow fails with Python 3.13:

   ```bash
   sudo apt install python3.10 python3.10-venv
   python3.10 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

4. **Verify Dataset** Ensure `Employee_Dataset.csv` is present with columns:

   - `Age`, `Experience`, `Education`, `Department`, `Location`, `Company Type`, `Performance Score`, `Last Hike %`, `Salary`

   Or generate using:

   ```bash
   python generate_dataset.py
   ```

5. **Run the App Locally**

   ```bash
   streamlit run app.py
   ```

   Open [http://localhost:8501](http://localhost:8501) in your browser.

---

### 🔹 Streamlit Cloud Deployment

1. **Push to GitHub**

   ```bash
   git add app.py requirements.txt Employee_Dataset.csv *.pkl *.keras
   git commit -m "Initial setup for Streamlit Cloud"
   git push origin main
   ```

2. **Deploy**

   - Log in to [Streamlit Cloud](https://streamlit.io/cloud).
   - Create a new app from your GitHub repository.
   - Set the main file to `app.py`.
   - Deploy. Streamlit will install dependencies from `requirements.txt`.

3. **Access App**

   - Open: `https://<your-app-name>.streamlit.app`

---

## 🧠 Usage

### 📥 Load Dataset

- Use default `Employee_Dataset.csv` or upload your own CSV.
- Required columns:
  ```
  Age (int), Experience (int), Education (str), Department (str),
  Location (str), Company Type (str), Performance Score (float),
  Last Hike % (float), Salary (float)
  ```

### 🛠️ Retrain Models

- Click **"Retrain Models"** in the sidebar.
- Models:
  - Linear Regression
  - Random Forest (with GridSearchCV)
  - Deep Learning (Dense → Dropout → BatchNorm)
- Time:
  - \~1–2 min for 1000 records
  - \~5–10 min for 5000 records (auto-downsampled)

### 📊 Make Predictions

- Enter employee features in the sidebar.
- Validates experience ≤ age - 16.
- Click **"Predict"** and select model.
- View prediction outputs per model.

### 📈 Visualizations

- R² Scores and MSE bar charts
- Feature importance (Random Forest)
- Actual vs. Predicted (scatter)
- Distributions of Salary, Age, Experience

### 🧾 Prediction History

- View stored predictions in a table.
- Export as `predictions.csv`
- Clear using **"Clear Prediction History"**

### 📂 Show Dataset

- Toggle "Show Dataset" to preview full CSV in-app.

---

## ⚙️ Performance Notes

| Dataset Size | Retrain Time | Accuracy         |
| ------------ | ------------ | ---------------- |
| \~1000 rows  | \~1–2 min    | R² ≅ 0.8–0.9     |
| \~5000 rows  | \~5–10 min   | Improved DL perf |

- Datasets >5000 are downsampled automatically.
- Sparse matrices used internally for performance.

---

## 🧯 Troubleshooting

### 🔻 IndexError

- Check logs in Streamlit Cloud ("Manage app")
- Run locally to debug:
  ```bash
  python -c "import pandas as pd; df = pd.read_csv('Employee_Dataset.csv'); print(df.columns); print(len(df))"
  ```

### 🔻 ModuleNotFoundError: joblib

- Ensure `joblib` is listed in `requirements.txt`
- Redeploy or run:
  ```bash
  pip install joblib
  ```

### 🔻 TensorFlow Compatibility

- Python 3.13 has issues with `tensorflow==2.16.2`
- Use Python 3.10 or update to compatible TensorFlow version.

### 🔻 Slow Performance

- Retrain locally for large datasets (>5000 records).
- Upgrade Streamlit Cloud tier for higher compute.

### 🔻 FileNotFoundError

- Ensure all model files (`.pkl`, `.keras`) and dataset are present.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m "Add feature"`
4. Push: `git push origin feature-name`
5. Open a pull request

---
## 🔗 Live Link: https://salary-prediction-ml-lthstozqbskjffj7mnfm6d.streamlit.app/


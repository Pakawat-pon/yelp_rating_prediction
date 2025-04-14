# 🌟 Yelp Business Rating Prediction App

This project predicts the Yelp rating for a new business using a simple Random Forest model. The system is divided into two parts:

1. **Backend**: Prepares data and trains a machine learning model.
2. **Frontend**: A Streamlit app for user interaction and real-time predictions.

> **Disclaimer**: This project is for academic purposes only.  
> Data is sourced from the Yelp Open Dataset.  
> The ML model is a basic Random Forest without fine-tuning.

---

## 📁 Workflow Overview

### 1️⃣ `Final_rating_prediction_app_backend.ipynb`

This notebook is responsible for:
- Importing and cleaning data
- Performing feature engineering
- Training a Random Forest model
- Predicting on new sample data
- Exporting the results as a `.parquet` file
- Saving the trained model with `joblib`

📌 **Code Breakdown**:
- `pandas`, `numpy`: Data manipulation
- `sklearn.ensemble.RandomForestRegressor`: Model used
- `nltk` or similar: Sentiment score generation (if applicable)
- Outputs:
  - `business_predictions.parquet`: Contains predictions
  - `model/random_forest_model.joblib`: Serialized ML model

---

### 2️⃣ `app.py`

This is the **Streamlit frontend** that:
- Loads the trained model
- Accepts user input:
  - Business city
  - Category
  - Sentiment score
  - Reviews/year
  - Check-ins/year
  - Attributes
- Applies preprocessing (e.g., label encoding if needed)
- Predicts and displays the expected star rating

📌 **Key Components**:
- `streamlit`: UI interface
- `joblib`: Load saved model
- `pandas`: For feature input and formatting
- Result: Predicted star rating shown interactively

---

## ⚙️ Getting Started

### Step 1: Clone the repo

```bash
git clone https://github.com/Pakawat-pon/Pakawat.pon.git
cd yelp_rating_prediction
```
### Step 2: Set up environment
pip install -r requirements.txt

### Step 3: Run the backend notebook (only once)
jupyter notebook Final_rating_prediction_app_backend.ipynb

### Step 4: Launch the Streamlit app from your terminal
streamlit run app.py

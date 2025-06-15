# Cardiovascular Diseases Prediction App

This project is a web-based application for predicting the risk of cardiovascular diseases using patient health data. It leverages a Logistic Regression model trained on real-world health datasets.

## Project Structure

```
.gitattributes
app.py
cardio_cleaned.csv
cardio_train.csv
code.ipynb
logistic_model.pkl
README.md
scaler.pkl
```

### File Descriptions

- **[app.py](app.py)**  
  The main Streamlit application. Loads the trained model and scaler, takes user input (age, height, weight, blood pressure, cholesterol, glucose, smoking, alcohol, activity), preprocesses the data, and predicts cardiovascular disease risk. Displays results and visualizations.

- **[cardio_cleaned.csv](cardio_cleaned.csv)**  
  Cleaned dataset used for model training and EDA. Contains patient health records with features such as age, gender, height, weight, blood pressure, cholesterol, glucose, smoking, alcohol, activity, and target label (`cardio`).

- **[cardio_train.csv](cardio_train.csv)**  
  The original/raw training dataset before cleaning and preprocessing.

- **[code.ipynb](code.ipynb)**  
  Jupyter notebook containing the full data science workflow: data cleaning, exploratory data analysis, feature engineering, model training, evaluation, and saving the trained model and scaler.

- **[logistic_model.pkl](logistic_model.pkl)**  
  Serialized Logistic Regression model trained on the cleaned dataset. Loaded by the app for making predictions.

- **[scaler.pkl](scaler.pkl)**  
  Serialized `StandardScaler` object used to scale input features before prediction. Ensures consistency between training and inference.

- **README.md**  
  Project documentation and instructions.

- **.gitattributes**  
  Git configuration file for handling line endings and file attributes.

## How to Run

1. **Start the app:**
   ```sh
   streamlit run app.py
   ```

2. **Interact with the dashboard:**  
   Enter patient data in the sidebar to get a cardiovascular disease risk prediction.

## Model Details

- **Algorithm:** Logistic Regression
- **Features:** Age, gender, height, weight, systolic/diastolic BP, cholesterol, glucose, smoking, alcohol, activity
- **Preprocessing:** Standard scaling

## Data Sources

- `cardio_train.csv`: Raw data
- `cardio_cleaned.csv`: Cleaned and processed data used for modeling

---

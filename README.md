# 🚗 Electric Vehicle Price Prediction Using Ridge Regression

## 📌 Project Overview

This project focuses on **predicting the price of electric vehicles (EVs) in India using Machine Learning**.

The notebook uses an EV car dataset containing information about different electric vehicle models, including their brand, model, range, power, and battery capacity.

A **Ridge Regression** model is used to predict the vehicle price. The project also applies preprocessing techniques such as **One-Hot Encoding** for categorical variables and **Standard Scaling** for numerical variables.

## 🎯 Objectives

The main objectives of this project are:

* Load and inspect the EV dataset.
* Understand the structure of the dataset.
* Check for missing values.
* Identify unique EV models.
* Separate input features and the target variable.
* Preprocess categorical and numerical features.
* Split the dataset into training and testing sets.
* Build a Ridge Regression model.
* Test different Ridge regularization values (`alpha`).
* Evaluate model performance using regression metrics.

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Jupyter Notebook**

## 🤖 Machine Learning Algorithm

### Ridge Regression

The project uses **Ridge Regression** for predicting EV prices.

Ridge Regression is a linear regression technique that includes regularization to reduce the effect of overly large model coefficients.

Different values of `alpha` are tested:

```text
0.01
0.1
1
10
100
```

This allows the model performance to be evaluated under different regularization strengths.

## 📂 Dataset

The dataset used in this project is:

```text
ev_car_India_dataset - ev_car_India_dataset.csv
```

The dataset contains information about electric vehicles available in India.

### Features Used

The model uses the following features:

| Feature   | Type        | Description                   |
| --------- | ----------- | ----------------------------- |
| `Brand`   | Categorical | EV manufacturer/brand         |
| `Model`   | Categorical | EV model                      |
| `Range`   | Numerical   | Driving range of the vehicle  |
| `Power`   | Numerical   | Power specification           |
| `Battery` | Numerical   | Battery capacity              |
| `Price`   | Target      | Price of the electric vehicle |

## 🔍 Project Workflow

The project follows these main steps:

```text
Dataset
   ↓
Data Inspection
   ↓
Missing Value Check
   ↓
Feature & Target Separation
   ↓
Categorical & Numerical Feature Identification
   ↓
Data Preprocessing
   ↓
Train-Test Split
   ↓
Ridge Regression
   ↓
Model Prediction
   ↓
Performance Evaluation
```

## 1. 📥 Importing Libraries

The notebook imports NumPy, Pandas, Matplotlib, Seaborn, and required Scikit-learn modules.

```python
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
```

Machine learning and evaluation libraries are also imported:

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

## 2. 📊 Loading the Dataset

The EV dataset is loaded using Pandas:

```python
df = pd.read_csv(
    "/content/ev_car_India_dataset - ev_car_India_dataset.csv"
)
```

The first few records are viewed using:

```python
df.head()
```

## 3. 🔎 Dataset Inspection

The notebook examines the dataset using:

```python
df.info()
df.shape
```

It also checks the unique EV models:

```python
df['Model'].unique()
```

## 4. 🧹 Missing Value Check

Missing values are checked using:

```python
df.isnull().sum()
```

This helps identify whether any features contain missing data before model training.

## 5. 🎯 Feature and Target Selection

The target variable is:

```text
Price
```

The input features are separated from the target:

```python
X = df.drop("Price", axis=1)
y = df["Price"]
```

Therefore:

* **X** → Input features
* **y** → EV Price

## 6. 🔤 Feature Classification

The categorical features are:

```python
categorical = ["Brand", "Model"]
```

The numerical features are:

```python
numerical = ["Range", "Power", "Battery"]
```

## 7. ⚙️ Data Preprocessing

A `ColumnTransformer` is used to preprocess the features.

### Categorical Features

`Brand` and `Model` are converted into numerical representations using **One-Hot Encoding**.

```python
OneHotEncoder(handle_unknown="ignore")
```

### Numerical Features

`Range`, `Power`, and `Battery` are standardized using:

```python
StandardScaler()
```

The preprocessing pipeline is:

```python
preprocessor = ColumnTransformer([
    (
        "categorical",
        OneHotEncoder(handle_unknown="ignore"),
        categorical
    ),
    (
        "numerical",
        StandardScaler(),
        numerical
    )
])
```

## 8. ✂️ Train-Test Split

The dataset is divided into training and testing data.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

The dataset is split into:

* **80% Training Data**
* **20% Testing Data**

## 9. 🤖 Ridge Regression Model

A machine learning pipeline is created by combining preprocessing and Ridge Regression.

```python
model = Pipeline([
    ("preprocessing", preprocessor),
    ("ridge", Ridge(alpha=alpha))
])
```

The model is trained using:

```python
model.fit(X_train, y_train)
```

## 10. 🔧 Alpha Values

The notebook evaluates Ridge Regression using different values of `alpha`:

```python
alphas = [0.01, 0.1, 1, 10, 100]
```

For each alpha value, the model is trained and predictions are generated.

This allows comparison of model performance for different regularization strengths.

## 11. 📈 Prediction

Predictions are generated for both training and testing datasets:

```python
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
```

These predictions are then compared with the actual EV prices.

## 12. 📊 Model Evaluation

The notebook evaluates the model using the following metrics:

### Mean Absolute Error (MAE)

```python
mean_absolute_error()
```

MAE measures the average absolute difference between actual and predicted prices.

### R² Score

```python
r2_score()
```

R² measures how well the model explains the variation in the target variable.

### RMSE-related Calculation

The notebook also calculates:

```python
np.sqrt(mean_absolute_error())
```

and stores the resulting value as `train_rmae` and `test_rmae`.

> Note: In the notebook this calculation is based on the square root of MAE rather than the standard RMSE formula.

## 📋 Results

The results for the different Ridge `alpha` values are stored in a Pandas DataFrame.

The result table contains:

| Column       | Description                                     |
| ------------ | ----------------------------------------------- |
| `alpha`      | Ridge regularization value                      |
| `train_mae`  | Training Mean Absolute Error                    |
| `train_rmae` | Calculated training metric used in the notebook |
| `train_r2`   | Training R² score                               |
| `test_mae`   | Testing Mean Absolute Error                     |
| `test_rmae`  | Calculated testing metric used in the notebook  |
| `test_r2`    | Testing R² score                                |

The final results are displayed using:

```python
print("\nRidge Regression")
print(results)
```

## 📌 Key Analysis Areas

The project covers:

* EV dataset inspection
* Data preprocessing
* Missing-value checking
* Categorical feature encoding
* Numerical feature scaling
* Train-test splitting
* Ridge Regression
* Hyperparameter comparison using alpha
* Price prediction
* MAE evaluation
* R² evaluation
* Training and testing performance comparison

## 📁 Project Structure

```text
Electric-Vehicle-Price-Prediction/
│
├── EV_Price_Prediction.ipynb
├── ev_car_India_dataset.csv
└── README.md
```

## 🚀 How to Run the Project

### Step 1: Install Required Libraries

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

### Step 2: Open Jupyter Notebook

```bash
jupyter notebook
```

### Step 3: Add the Dataset

Place the EV dataset in the project directory.

Update the dataset path in the notebook if required:

```python
df = pd.read_csv("ev_car_India_dataset.csv")
```

### Step 4: Run the Notebook

Run the cells sequentially to:

1. Load the dataset.
2. Inspect the data.
3. Preprocess the features.
4. Split the data.
5. Train Ridge Regression models.
6. Generate predictions.
7. Evaluate the models.
8. Compare the results for different alpha values.

## 🎓 Skills Demonstrated

This project demonstrates practical knowledge of:

* Python
* Pandas
* NumPy
* Machine Learning
* Scikit-learn
* Data Preprocessing
* One-Hot Encoding
* Feature Scaling
* Train-Test Split
* Ridge Regression
* Machine Learning Pipelines
* Hyperparameter Testing
* Regression Evaluation
* MAE
* R² Score

## 👩‍💻 Author

**Madhumidha**

# 🧠 Diabetes Progression Prediction using Artificial Neural Network

## 📌 Project Overview
This project focuses on modeling the progression of diabetes using an **Artificial Neural Network (ANN)**. The model is built using the Diabetes dataset from the `sklearn` library and aims to help understand how different clinical factors influence disease progression.

## 🎯 Objective
To design and evaluate an **ANN-based regression model** that predicts diabetes progression based on multiple medical features, and to analyze how model improvements affect prediction performance.

## 📂 Dataset

**Source:** `sklearn.datasets.load_diabetes`  

**Description:**  
The dataset contains **442 samples** with **10 normalized medical features** such as age, BMI, blood pressure, and serum measurements.  

**Target:**  
A quantitative measure of diabetes disease progression one year after baseline.

## 🛠️ Technologies Used
- Python  
- NumPy  
- Pandas  
- Matplotlib & Seaborn  
- Scikit-learn  
- TensorFlow / Keras  
- Jupyter Notebook  

## 🔍 Project Workflow

### 1. Data Loading & Preprocessing
- Loaded the diabetes dataset from sklearn
- Checked for missing values
- Applied feature normalization using `StandardScaler`

### 2. Exploratory Data Analysis (EDA)
- Visualized feature distributions and target variable
- Analyzed feature–target relationships using scatter plots
- Used correlation matrix to identify important features

### 3. Outlier Detection
- Detected outliers using boxplots and the IQR method
- Outliers were retained as they may represent genuine clinical cases

### 4. Model Building
- Designed an ANN with multiple hidden layers
- Used **ReLU** activation for hidden layers
- Used a single neuron output layer for regression

### 5. Model Training
- Split data into training and testing sets
- Trained the model using **Adam optimizer** and **MSE loss**
- Used validation split to monitor training performance

### 6. Model Evaluation
Evaluated performance using:
- Mean Squared Error (MSE)  
- R² Score  

### 7. Model Improvement
- Increased network depth and number of neurons
- Trained for more epochs
- Observed improved MSE and R² score after tuning

## 📊 Results

| Metric | Before Improvement | After Improvement |
|--------|------------------|-----------------|
| MSE    | 2954.96           | 2750.10          |
| R²     | 0.44              | 0.48             |

> The improved model shows better prediction accuracy and explains more variance in diabetes progression.

## ✅ Conclusion
The ANN model successfully learned the relationship between clinical features and diabetes progression. Model performance improved after architectural and hyperparameter tuning, demonstrating the effectiveness of deep learning techniques for **medical regression problems**.

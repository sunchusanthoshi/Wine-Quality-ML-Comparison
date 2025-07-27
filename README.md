# 🍷 Wine Quality Prediction: A Comparative Analysis of ML Models

## 📌 Project Overview

This project aims to **predict the quality of wine** based on its physicochemical properties using three different machine learning models:

- Logistic Regression  
- Support Vector Machine (SVM)  
- Neural Networks (Multilayer Perceptron)

We evaluate and compare each model’s performance using a consistent pipeline including data preprocessing, class balancing (SMOTE), scaling, and performance metrics (accuracy, F1-score, confusion matrix, etc.).

## 🧪 Dataset Description

- **Source:** [Wine Quality Dataset on Kaggle](https://www.kaggle.com/datasets/rajyellow46/wine-quality)
- **Samples:** 6,497 (Red and White Portuguese wines)
- **Features:**  
  Fixed Acidity, Volatile Acidity, Citric Acid, Residual Sugar, Chlorides, Free/Total Sulfur Dioxide, Density, pH, Sulphates, Alcohol  
- **Target:** `quality` (integer from 0 to 10)


## ⚙️ Preprocessing Steps

- Removed non-predictive `Id` column  
- Scaled features using `StandardScaler`  
- Balanced the dataset with `SMOTE`  
- Split into 80% training and 20% testing sets  
- Applied zero-indexing for categorical labels as needed

## 🤖 Models and Results

### 1. **Logistic Regression**
- Accuracy: **45%**
- Moderate precision/recall
- Shows limitations with multi-class prediction on this dataset

### 2. **Support Vector Machine (SVM)**
- Accuracy: **64.48%**
- Linear kernel used
- Better performance than logistic regression but not optimal

### 3. **Neural Network (MLP)**
- Accuracy: **76.55%** ✅ *Best model*
- Trained over 50 epochs
- Clear learning curve with improving loss and accuracy
- Most effective in capturing complex feature interactions

## 📊 Visualizations

- **Histograms** of all input features  
- **Heatmap** for feature correlation  
- **Confusion matrices** for all models  
- **Training/Validation accuracy & loss curves** for neural network


## 🧠 Conclusion

- **Neural Network** significantly outperformed the other models in terms of accuracy and generalization  
- Traditional models like Logistic Regression and SVM still provide interpretability and insight  
- **Feature importance** highlights the impact of alcohol, sulphates, and acidity on wine quality


## 🛠️ Tools and Libraries

- Python  
- Pandas, NumPy, Scikit-learn  
- TensorFlow / Keras  
- Matplotlib, Seaborn  
- imbalanced-learn (SMOTE, RandomOverSampler)

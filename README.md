# 🍷 Wine Quality Prediction using Machine Learning and Deep Learning

This project aims to predict the quality of red wine based on physicochemical features using various classification algorithms. The goal is to classify wine as **good** or **not good** based on its attributes like acidity, alcohol, pH, etc.

---

## 📌 Project Description

The dataset contains 1599 samples of red wine, each with 11 numerical features. We formulated this as a **binary classification task**:
- **Good wine**: Quality score ≥ 7
- **Not good**: Quality score < 7

A range of models were used and compared based on their accuracy, precision, recall, and F1 score.

---

## 🧠 Models Used

The following algorithms were implemented:

- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Naive Bayes
- **XGBoost** (Extreme Gradient Boosting)
- **Artificial Neural Network (ANN)** using Keras
- **Voting Classifier** (Ensemble of top models)

---

## ⚙️ Tools & Libraries

- Python
- scikit-learn
- XGBoost
- Keras / TensorFlow
- NumPy & Pandas
- Matplotlib & Seaborn (for data visualization)

---

## 🧪 Dataset

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- **Samples**: 1599 red wine instances
- **Features**: 11 physicochemical inputs + 1 quality score (target)

---

## 📈 Feature Scaling

Before training, we applied **StandardScaler** to normalize all features so that they have a mean of 0 and standard deviation of 1. This improves convergence and model performance, especially for ANN and distance-based models like KNN.

---

## 📊 Evaluation Metrics

Each model was evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

We also used **GridSearchCV** for hyperparameter tuning to optimize model performance.

---

## 🔍 Feature Importance & Visualization

Random Forest and XGBoost feature importances were visualized to understand the most impactful features on wine quality. Alcohol, volatile acidity, and sulphates were among the top contributors.

---

## 📂 Project Structure

```bash
├── WinePrediction.ipynb        # Main Jupyter Notebook
├── README.md                   # Project overview and instructions
├── requirements.txt            # Dependencies (optional)

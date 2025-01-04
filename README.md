# **Diabetes Prediction**

This repository contains a Jupyter Notebook for predicting diabetes using machine learning techniques. The dataset used in this project was sourced from a publicly available diabetes dataset.

---

## **Overview**

Diabetes is a chronic condition that affects millions of people worldwide. Early detection and prediction can help in managing and mitigating its effects. This project demonstrates how to use machine learning models to predict whether an individual has diabetes based on various health metrics.

The dataset includes features such as glucose levels, blood pressure, BMI, and other diagnostic measurements. The target variable (`Outcome`) indicates whether the individual has diabetes (1) or not (0).

---

## **Dataset**

- **Source**: The dataset appears to be related to the [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database), commonly used for classification tasks.
- **Features**:
  - `Pregnancies`: Number of times pregnant.
  - `Glucose`: Plasma glucose concentration.
  - `BloodPressure`: Diastolic blood pressure (mm Hg).
  - `SkinThickness`: Triceps skinfold thickness (mm).
  - `Insulin`: 2-Hour serum insulin (mu U/ml).
  - `BMI`: Body mass index (weight in kg/(height in m)^2).
  - `DiabetesPedigreeFunction`: Diabetes pedigree function (a measure of genetic influence).
  - `Age`: Age of the individual.
- **Target Variable**:
  - `Outcome`: Indicates whether the individual has diabetes (1 = Yes, 0 = No).

---

## **Project Workflow**

1. **Data Loading**:
   - The dataset (`diabetes.csv`) is loaded into a Pandas DataFrame.
2. **Data Preprocessing**:
   - StandardScaler is used to normalize numerical features for better model performance.
3. **Model Training**:
   - Two machine learning models are trained:
     - Logistic Regression
     - Support Vector Machine (SVM) with a linear kernel
   - The dataset is split into training and testing sets using `train_test_split`.
4. **Model Evaluation**:
   - Accuracy scores are calculated for both models to evaluate their performance.

---

## **Dependencies**

To run this project, you need the following Python libraries:

- pandas
- numpy
- scikit-learn

You can install these dependencies using pip:

```bash
pip install pandas numpy scikit-learn
```

---

## **How to Run**

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/DiabetesPrediction.git
   cd DiabetesPrediction
   ```

2. Ensure that the dataset file (`diabetes.csv`) is in the same directory as the notebook.

3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Diabetes-Pred.ipynb
   ```

4. Run all cells in the notebook to execute the code.

---

## **Results**

The machine learning models provide accuracy scores that indicate their performance in predicting diabetes. Logistic Regression and SVM are evaluated, and their results can be compared for further insights.

---

## **Acknowledgments**

- The dataset was sourced from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) or similar public repositories.
- Special thanks to the original contributors of the Pima Indians Diabetes Database.

---

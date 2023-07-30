# Liver Disease Prediction Model using Ensemble Learning
This repository contains code for a Liver Disease Prediction Model using Ensemble Learning. The model has achieved an impressive 100% accuracy in predicting liver disease based on a set of input attributes. The project also provides a web application using Streamlit, where users can input their attributes and get predictions for liver disease. Additionally, users can compare the results of the proposed ensemble model with other selected classifiers.

# Live Demo
A live demo of the Liver Disease Prediction web application can be accessed at the following link: https://liver-disease-prediction-using-ensemble.streamlit.app/

# How to Use the Web Application
Access the live demo link provided above.

Fill in the required input attributes, including age, gender, total bilirubin (TB), direct bilirubin (DB), alkaline phosphatase (ALP), alamine aminotransferase (SGPT), aspartate aminotransferase (SGOT), total proteins (TP), albumin (ALB), and the albumin and globulin ratio (A/G Ratio).

Choose the classifier models from the available list to compare with the proposed ensemble model.

Click the "Submit" button to get the predictions and performance metrics for the selected models.

# Models Available for Comparison
The web application allows users to select and compare the following classifier models:

1. Random Forest
2. Naïve Bayes
3. Logistic Regression
4. K-Nearest Neighbors
5. Decision Tree
6. Gradient Boosting
7. Support Vector Machine
8. LightGBM
9. XGBoost
10. Multilayer Perceptron (Artificial Neural Network)

# Note
It is essential to remember that achieving a 100% accuracy on any real-world dataset may indicate potential issues such as overfitting or data leakage. While the model may perform well on the current dataset, it is essential to assess its performance on unseen data and conduct a thorough evaluation before deploying it in a real-world scenario.

# Dataset
The liver disease prediction model is trained on a dataset named 'Liver.csv,' which contains the necessary input features and their corresponding target labels (1 for liver disease and 0 for no liver disease). The data preprocessing steps include dropping any rows with missing values and encoding categorical variables.

# Installation
To run the application locally or contribute to the project, follow these steps:

1. Clone the repository:
```
git clone https://github.com/SaifSunny/Liver-Disease-Prediction-using-Ensemble.git
```
2. Install the required libraries:
```
pip install streamlit pandas numpy matplotlib scikit-learn xgboost lightgbm
```
3. Run the Streamlit app:
```
streamlit run main.py
```
# Contributions
Contributions to the project are welcome! If you have any issues or suggestions, feel free to submit a pull request or open an issue.

# License
The project is under the MIT License.


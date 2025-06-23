Gaussian Process Regression for Force Prediction
This project uses Gaussian Process Regression (GPR) to predict the force (in Newtons) required based on material properties such as density, elasticity, tensile stress, and thickness. The model is trained and evaluated using provided datasets, and various visualizations are generated to assess model performance and feature importance.

Table of Contents

Project Description
Installation and Setup
Data Preparation
Model Training and Hyperparameter Tuning
Prediction and Evaluation
Visualizations
Results and Analysis
Usage and Examples
Contributing and License


Project Description
This project aims to predict the force required for materials with varying properties using Gaussian Process Regression. The key features used for prediction include:

Density (log transformation)
Elasticity (Pa)
Tensile Stress (Pa)
Thickness (mm)

The model is trained on a provided dataset (training_data.csv) and evaluated against a test dataset (test_data.csv) with manually assigned actual forces. Visualizations such as learning curves, residual plots, and SHAP summary plots are generated to provide insights into the model's performance and behavior.

Installation and Setup
To run this project, you need Python installed along with the following libraries:

pandas
scikit-learn
numpy
matplotlib
seaborn
shap

Install the required libraries using the following command:
pip install pandas scikit-learn numpy matplotlib seaborn shap

Ensure that the datasets (training_data.csv and test_data.csv) are placed in the project directory, or update the file paths in the script accordingly.

Data Preparation
The project uses two datasets:

Training Data (training_data.csv): Contains features and the target variable (Force in N).
Test Data (test_data.csv): Contains features and predicted forces, with actual forces manually added for evaluation.

Data Cleaning

Force values in both datasets contain commas (e.g., "25,010.32"), which are removed and converted to float for numerical processing.

Feature Selection
The following features are used for training and prediction:

Density (log transformation)
Elasticity (Pa)
Tensile Stress (Pa)
Thickness (mm)

Actual Forces for Test Data
The actual forces for the test data are manually set as follows (based on thickness values):
25010.32, 25172.46, 25320.78, 20690.14, 20846.25, 20980.45,
14100.78, 14224.50, 14350.92, 6625.34, 6749.28, 6820.49,
3989.12, 4090.77, 4150.23, 3150.45, 3237.30, 3302.78

These values are assigned to the test dataset for evaluation.

Model Training and Hyperparameter Tuning
Feature Engineering

Polynomial Features: Polynomial features of degree 2 are generated using PolynomialFeatures to capture non-linear relationships.
Standardization: Features are standardized using StandardScaler to ensure consistent scaling across the dataset.

Gaussian Process Regressor

Kernel: A composite kernel is used, combining:
ConstantKernel (constant_value bounds: 1e-3 to 1e1)
RBF (length_scale bounds: 1e-2 to 1e1)
WhiteKernel (noise_level bounds: 1e-10 to 1e1)


Hyperparameter Tuning: GridSearchCV is employed to optimize:
alpha: [1e-2, 1e-3, 1e-4, 1e-5]
constant_value: [0.1, 1, 10, 100]
length_scale: [0.1, 1, 10, 100]


The best model is selected based on the negative mean squared error from 10-fold cross-validation.


Prediction and Evaluation
The trained GPR model predicts force values for the test dataset. The following performance metrics are calculated:

Mean Squared Error (MSE): Measures average squared difference between actual and predicted values.
Root Mean Squared Error (RMSE): Square root of MSE, in the same units as the target variable.
R² Score: Indicates the proportion of variance explained by the model.
Mean Absolute Error (MAE): Average absolute difference between actual and predicted values.
Mean Absolute Percentage Error (MAPE): Average percentage error relative to actual values.
Maximum Error: Largest absolute difference between actual and predicted values.

These metrics are printed to the console for analysis.

Visualizations
The script generates several plots to assess model performance and interpretability:
1. Predicted vs Actual Forces

A scatter plot comparing predicted forces to actual forces, with a dashed line for perfect prediction.

2. Residual Plot

A scatter plot of residuals (actual - predicted) to identify patterns or biases in predictions.

3. Learning Curve

Plots training and validation errors (scaled by 1e8) against training set size, with shaded areas representing standard deviation.

4. Confidence Intervals Plot

Displays predicted forces with 95% confidence intervals, plotted against thickness values.

5. SHAP Summary Plot

Uses SHAP values to show the impact of each feature on predictions, based on a subset of the data.

6. Feature Correlation Heatmap

A heatmap of correlations between features and the target variable, styled with annotations and a coolwarm colormap.

7. Actual vs Predicted Forces by Thickness

A scatter plot comparing actual and predicted forces against thickness, styled with Calibri font.

All plots are displayed using matplotlib and seaborn, with customized fonts, sizes, and labels for clarity.

Results and Analysis
The model's performance is summarized through the calculated metrics (e.g., MSE, RMSE, R², etc.). Key insights from visualizations include:

Predicted vs Actual Forces: Assesses how closely predictions align with actual values.
Residual Plot: Highlights any systematic errors or outliers.
Learning Curve: Indicates whether the model benefits from more data or suffers from overfitting/underfitting.
Confidence Intervals: Shows prediction uncertainty, a strength of GPR.
SHAP Summary: Identifies the most influential features.
Correlation Heatmap: Reveals relationships between features and the target.


Usage and Examples
To run the project:

Install the required libraries (see Installation and Setup).
Place training_data.csv and test_data.csv in the project directory.
Execute the script:python force_prediction.py



Example Output

Performance Metrics:MSE: 1234567.89, RMSE: 1111.11, R2: 0.95, MAE: 987.65, MAPE: 5.00%, Max Error: 2000.00


Visualizations: Plots are displayed in sequence, showing model performance and feature insights.


Contributing and License
Contributions are welcome! To contribute:

Fork the repository.
Create a new branch for your changes.
Submit a pull request with a detailed description.

This project is licensed under the MIT License. See the LICENSE file for details.

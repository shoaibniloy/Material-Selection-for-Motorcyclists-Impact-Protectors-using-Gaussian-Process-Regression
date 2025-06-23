
## **Material-Selection-for-Motorcyclists-Impact-Protectors-using-Gaussian-Process-Regression**

### Overview

This repository demonstrates the use of **Gaussian Process Regression (GPR)** for the **material selection** process in **motorcyclists' impact protectors**. The goal is to predict the force exerted on a material based on various material properties such as **Density**, **Elasticity**, **Tensile Stress**, and **Thickness**. The model uses a set of input features (material properties) to predict the force experienced by the material under impact conditions. This model aids in selecting optimal materials that would provide the best protection for motorcyclists.

The project includes data preprocessing, feature engineering, model training using Gaussian Process Regression, performance evaluation, and visualization. The notebook also showcases how to assess the model’s generalization capabilities through learning curves and evaluation metrics.

### Key Features

* **Gaussian Process Regressor**: Utilized for capturing complex relationships between input material properties and force values.
* **Polynomial Features**: Used to extend the input features and allow the model to learn non-linear relationships.
* **Model Evaluation**: Evaluation metrics such as **Mean Squared Error (MSE)**, **Root Mean Squared Error (RMSE)**, **R²**, and **Mean Absolute Error (MAE)** are used to assess model accuracy.
* **Learning Curves**: Visualizations to understand the model’s performance over different training sizes.

---

### Repository Structure

```plaintext
Material-Selection-for-Motorcyclists-Impact-Protectors-using-Gaussian-Process-Regression/
│
├── training_data.csv              # The training dataset, including material properties and corresponding force values.
├── test_data.csv                  # The test dataset, which includes material properties and predicted force values.
└── Final_code.ipynb               # Jupyter Notebook containing all steps: data preprocessing, modeling, and evaluation.
```

---

### Installation Instructions

To run this project on your local machine, follow these steps:

1. **Clone the repository** to your local machine:

   ```bash
   git clone https://github.com/shoaibniloy/Material-Selection-for-Motorcyclists-Impact-Protectors-using-Gaussian-Process-Regression.git
   cd Material-Selection-for-Motorcyclists-Impact-Protectors-using-Gaussian-Process-Regression
   ```

2. **Create a Python virtual environment** to manage dependencies:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   If the `requirements.txt` file is unavailable, install the dependencies manually:

   ```bash
   pip install pandas scikit-learn matplotlib numpy shap seaborn
   ```

4. **Launch the Jupyter Notebook** to start working with the code:

   ```bash
   jupyter notebook
   ```

---

### Data Description

This repository includes two CSV files that are crucial for training and evaluating the model:

* **`training_data.csv`**:
  This dataset is used to train the Gaussian Process model. It contains material properties and their corresponding force values. The columns include:

  * `Material`: The type of material (e.g., different polymers, composites).
  * `Yield Strength (Pa)`: The material’s yield strength, measured in Pascals.
  * `Density (log transformation)`: The log-transformed density value of the material.
  * `Elasticity (Pa)`: The elasticity or Young's modulus of the material in Pascals.
  * `Tensile Stress (Pa)`: The tensile stress capacity of the material in Pascals.
  * `Thickness (mm)`: Thickness of the material in millimeters.
  * `Force (N)`: The corresponding force exerted on the material in Newtons. This is the target variable for prediction.

* **`test_data.csv`**:
  The test dataset, which has similar material properties to the training data, is used for evaluation. It contains:

  * `Material`: The type of material.
  * `Yield Strength (Pa)`: The yield strength in Pascals.
  * `Density (log transformation)`: The log-transformed density.
  * `Elasticity (Pa)`: Elasticity in Pascals.
  * `Tensile Stress (Pa)`: Tensile stress.
  * `Thickness (mm)`: Thickness in millimeters.
  * `Predicted Force (N)`: Predicted force values, which are compared against actual forces.
  * `Actual Force`: The true force values exerted on the material, used to evaluate the model's performance.

---

### Workflow

The process is broken down into several stages in the **`Final_code.ipynb`** notebook, which includes the following steps:

#### 1. **Data Preprocessing**

* **Loading Data**: The training and test datasets are loaded using `pandas.read_csv()`.
* **Cleaning Data**: The `Force (N)` columns contain commas (e.g., `23,450`), which are removed to ensure correct parsing of numeric values. The columns are then converted to `float` data type.
* **Manual Force Assignment**: For the test dataset, actual force values are manually assigned based on the thickness of the material.

#### 2. **Feature Engineering**

* **Polynomial Features**: Polynomial features of degree 2 are generated using `PolynomialFeatures()` to capture higher-order interactions between the material properties.
* **Feature Standardization**: Features are standardized using `StandardScaler()` to improve the convergence of the model.

#### 3. **Model Training**

* **Gaussian Process Regression (GPR)**: The core of the model, using a combination of **Radial Basis Function (RBF)** and **White Kernel** for noise modeling. The GPR model is fitted to the training data.
* **Hyperparameter Tuning**: A `GridSearchCV` is employed for hyperparameter optimization, though this step can be modified or extended as needed.

#### 4. **Model Evaluation**

* After training the model, predictions are made on the test data.
* The model's performance is evaluated using:

  * **Mean Squared Error (MSE)**
  * **Root Mean Squared Error (RMSE)**
  * **R² (coefficient of determination)**
  * **Mean Absolute Error (MAE)**
* These metrics help assess how well the model generalizes to unseen data.

#### 5. **Visualization**

* **Predicted vs Actual Force**: A scatter plot is created to compare the predicted force vs. the actual force values.
* **Learning Curves**: The learning curves plot the model’s error on both the training and validation sets as the training set size increases. This helps identify underfitting or overfitting.

These visualizations are essential for interpreting the model's behavior and performance.

---

### Example Usage

1. Clone the repository and install dependencies as described above.
2. Open the Jupyter notebook (`Final_code.ipynb`).
3. Execute the cells in the notebook to:

   * Load and clean the data.
   * Preprocess the features.
   * Train the Gaussian Process Regression model.
   * Evaluate model performance and visualize results.
4. Analyze the results:

   * Check the scatter plot for predicted vs. actual force.
   * Review the learning curve to evaluate model stability and generalization.

---

### Requirements

* Python 3.x
* Jupyter Notebook
* Libraries:

  * **pandas**: For data manipulation and preprocessing.
  * **scikit-learn**: For machine learning model training, evaluation, and metrics.
  * **matplotlib**: For creating visualizations such as learning curves and predicted vs. actual force.
  * **numpy**: For numerical operations.
  * **shap**: For model explainability (optional, depending on the notebook version).
  * **seaborn**: For additional visualizations (optional).

---

### License

This repository is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more information.

---

### Acknowledgments

* **Gaussian Process Regressor**: A powerful non-linear regression method used to model complex relationships in the data.
* **Scikit-learn**: A comprehensive machine learning library used for regression, model evaluation, and hyperparameter tuning.
* **Matplotlib & Seaborn**: Used for creating insightful visualizations to understand model performance.
* **Jupyter Notebook**: Provides an interactive environment to document and run the code step by step.


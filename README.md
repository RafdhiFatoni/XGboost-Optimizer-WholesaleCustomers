# XGBoost Optimizer for Wholesale Customer Segmentation ✨
<p align="center">
<a href="https://opensource.org/licenses/MIT">
<img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
</a>
<a href="https://www.python.org/">
<img src="https://img.shields.io/badge/Python-3.x-blue.svg" alt="Python 3.x">
</a>
<a href="https://github.com/psf/black">
<img src="https://img.shields.io/badge/Code%20Style-Black-000000.svg" alt="Code style: black">
</a>
<a href="https://github.com/RafdhiFatoni/XGboost-Optimizer-WholesaleCustomers/commits/main">
<img src="https://img.shields.io/github/last-commit/RafdhiFatoni/XGboost-Optimizer-WholesaleCustomers.svg" alt="Last Commit">
</a>
<br>
<img src="https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white" alt="Pandas">
<img src="https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white" alt="NumPy">
<img src="https://img.shields.io/badge/Scikit--Learn-F7931A?logo=scikit-learn&logoColor=white" alt="Scikit-learn">
<img src="https://img.shields.io/badge/XGBoost-006B00?logo=xgboost&logoColor=white" alt="XGBoost">
<img src="https://img.shields.io/badge/Jupyter-F37626?logo=Jupyter&logoColor=white" alt="Jupyter">
</p>

<p align="center">
A comprehensive machine learning project focused on classifying wholesale customers by leveraging the power of <b>XGBoost</b>. The primary goal is to perform hyperparameter tuning to find the optimal model for segmenting customers into 'Horeca' (Hotel/Restaurant/Café) or 'Retail' channels based on their annual spending habits.
</p>

## 🎯 About The Project
Understanding customer segments is vital for targeted marketing and business strategy. This project tackles this challenge by building a robust classification model. It walks through the entire data science pipeline, from data exploration and preprocessing to model building, and most importantly, performance optimization through systematic hyperparameter tuning of an XGBoost classifier.

## 📊 Dataset
The project utilizes the **"Wholesale customers"** dataset from the UCI Machine Learning Repository. It contains the annual spending data of 440 clients on diverse product categories.
### Attributes:
- `FRESH`: Annual spending on fresh products.
- `MILK`: Annual spending on milk products.
- `GROCERY`: Annual spending on grocery products.
- `FROZEN`: Annual spending on frozen products.
- `DETERGENTS_PAPER`: Annual spending on detergents and paper products.
- `DELICATESSEN`: Annual spending on delicatessen products.
- `CHANNEL`: Customer's Channel (1: Horeca, 2: Retail) - Target Variable.
- `REGION`: Customer's Region (1: Lisbon, 2: Oporto, 3: Other).

## 🛠️ Methodology
The project follows a structured approach to solve the classification problem:
1. **Exploratory Data Analysis (EDA):** Analyzing the dataset to understand feature distributions.
2. **Data Preprocessing:** Check missing valu and etc. , which is crucial for many ML algorithms.
3. **Data Engineering:** Preparing data for train the machine learning
4. **Model Building:** Implementing a XGBoost classifier.
5. **Hyperparameter Optimization:** Using techniques like GridSearchCV to find the best combination of hyperparameters (n_estimators, max_depth, learning_rate, etc.) for the XGBoost model.
6. **Model Evaluation:** Assessing the performance of the final, optimized model using metrics like Accuracy, Precision, Recall, F1-Score, and the Confusion Matrix.
7. **Feature Importance:** Analyze weight of feature from the model

## 🚀 Tech Stack
This project is built with Python and relies on the following major libraries:
1. **Python 3.x**
2. **NumPy:** For numerical operations.
3. **Pandas:** For data manipulation and analysis.
4. **Matplotlib & Seaborn:** For data visualization.
5. **Scikit-learn:** For data preprocessing, model evaluation, and cross-validation.
6. **XGBoost:** For building the high-performance gradient boosting model.
7. **Jupyter Notebook:** For interactive development and analysis.

## ⚙️ Getting Started
To get a local copy up and running, follow these simple steps.
### Prerequisites
Ensure you have Python 3 installed on your system.
### Installation
1. Clone the repository:
```
git clone https://github.com/RafdhiFatoni/XGboost-Optimizer-WholesaleCustomers.git
cd XGboost-Optimizer-WholesaleCustomers
```
2. Create and activate a virtual environment (recommended):
```
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```
3. Install the required packages:
```
pip install -r requirements.txt
```
### Usage
Once the setup is complete, you can explore the project:
1. Launch Jupyter Notebook:
```
jupyter notebook
```
2. Open the `main_model.ipynb` file and run the cells sequentially to see the entire workflow from data loading to model evaluation.

## 📈 Results
The final, optimized XGBoost model demonstrated a high level of accuracy in classifying customers. The Jupyter Notebook contains detailed outputs, including:
1. A feature importance plot highlighting the key drivers of customer segmentation.
2. A confusion matrix and classification report to showcase the model's predictive power.
3. A best model from optimizer, proving the effectiveness of optimization.

## 🤝 Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated.**
If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag _"enhancement"_.
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License
Distributed under the MIT License. See `LICENSE` file for more information.

## 🙏 Acknowledgements
- The Wholesale customers dataset is provided by the **UCI Machine Learning Repository.**
- Hat tip to anyone whose code was an inspiration.

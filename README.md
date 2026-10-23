# Machine Learning & Data Science Internship

**Organization:** Datheon  
**Domain:** Machine Learning & Data Science  
**Technologies:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  
**Project Type:** Supervised Machine Learning, Predictive Modeling, and Data Analysis

## 📌 Project Overview

This repository contains my work for the **Machine Learning & Data Science Internship at Datheon**. The objective is to develop practical machine learning skills by building predictive models, preprocessing datasets, performing feature engineering, evaluating algorithms, and visualizing model performance.

The internship tasks focus on real-world machine learning applications, including classification, house price prediction, customer churn prediction, and natural language processing (NLP).

Each completed task includes clean Python code, exploratory data analysis, model training, evaluation metrics, and visualizations wherever applicable.

## 🎯 Internship Objectives

- Understand and implement supervised machine learning algorithms.
- Perform data cleaning, preprocessing, and missing-value imputation.
- Apply feature scaling, encoding, and feature engineering techniques.
- Train and compare multiple machine learning models.
- Evaluate classification and regression models using appropriate metrics.
- Visualize model performance using plots, heatmaps, and evaluation curves.
- Develop well-documented and reproducible Jupyter Notebooks.
- Maintain project code and documentation using Git and GitHub.

## 🛠️ Technologies and Tools

| Technology | Purpose |
|---|---|
| Python | Programming and model development |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computations |
| Matplotlib | Data visualization |
| Seaborn | Statistical plots and heatmaps |
| Scikit-learn | Machine learning and evaluation |
| Jupyter Notebook | Interactive coding and documentation |
| Git and GitHub | Version control and project hosting |
| NLTK / Regular Expressions | Text preprocessing for NLP |
| XGBoost | Gradient boosting for classification |
| imbalanced-learn | Handling imbalanced datasets |

## 📂 Internship Tasks

### Task 1: Multi-Class Benchmark Classification Pipeline

**Difficulty:** Beginner

**Objective:** Build and compare supervised classification models using the Iris or Wine dataset.

**Implementation:**
- Load and inspect the dataset using Pandas.
- Perform exploratory data analysis (EDA).
- Visualize feature relationships using Seaborn pairplots.
- Separate features and target labels.
- Split the data into training and testing sets using an 80:20 ratio.
- Scale numerical features using `StandardScaler`.
- Train Logistic Regression and Decision Tree classifiers.
- Evaluate model performance using accuracy, precision, recall, and F1-score.
- Visualize confusion matrices using heatmaps.
- Create a decision-boundary visualization where applicable.

**Algorithms:**
- Logistic Regression
- Decision Tree Classifier

**Evaluation Metrics:**
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

**Dataset:** [Iris Dataset](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv)

**Deliverable:** A Jupyter Notebook containing data analysis, trained models, classification reports, and evaluation visualizations.

### Task 2: Real Estate Housing Price Prediction

**Difficulty:** Beginner

**Objective:** Predict median house values using regression algorithms and compare their predictive performance.

**Implementation:**
- Load and explore the California Housing dataset.
- Identify and handle missing values.
- Analyze feature correlations and potential multicollinearity.
- Create interaction features for selected numerical variables.
- Encode categorical features where necessary.
- Split the dataset into training and testing sets.
- Standardize numerical features.
- Train Ordinary Least Squares (OLS), Ridge, and Lasso regression models.
- Use `GridSearchCV` to tune Ridge and Lasso regularization parameters.
- Compare model performance using RMSE, MAE, and R².
- Visualize actual versus predicted prices and analyze residual errors.

**Algorithms:**
- Ordinary Least Squares (Linear Regression)
- Ridge Regression (L2 Regularization)
- Lasso Regression (L1 Regularization)

**Evaluation Metrics:**
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Coefficient of Determination (R²)

**Dataset:** [California Housing Dataset](https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv)

**Deliverable:** A Jupyter Notebook comparing regression models, tuned hyperparameters, evaluation metrics, and residual plots.

### Task 3: Predictive Customer Churn Classification

**Difficulty:** Intermediate

**Objective:** Predict customer churn and evaluate classification models on an imbalanced telecom customer dataset.

**Implementation:**
- Load and explore the telecom customer churn dataset.
- Convert `TotalCharges` into a numeric format.
- Handle missing values and duplicate records where appropriate.
- Encode categorical variables using `OneHotEncoder`.
- Split the data into training and testing sets.
- Address class imbalance using class weighting or SMOTE on the training data only.
- Train Random Forest and XGBoost classifiers.
- Apply cross-validation and model tuning where appropriate.
- Evaluate model performance using classification metrics and ROC-AUC.
- Plot ROC curves and confusion matrices.
- Identify important features influencing churn predictions.

**Algorithms:**
- Random Forest Classifier
- XGBoost Classifier

**Evaluation Metrics:**
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

**Dataset:** [Telco Customer Churn Dataset](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)

**Deliverable:** A Jupyter Notebook containing the churn prediction pipeline, evaluation charts, ROC curve, confusion matrix, and feature importance visualization.

### Task 4: NLP Sentiment Classification on Customer Reviews

**Difficulty:** Intermediate

**Objective:** Develop a natural language processing pipeline to classify customer reviews as positive or negative.

**Implementation:**
- Load and inspect the sentiment review dataset.
- Clean text by removing HTML tags and unnecessary punctuation.
- Normalize text through lowercasing and appropriate tokenization.
- Apply stopword removal and lemmatization where appropriate.
- Convert text into numerical features using `TfidfVectorizer`.
- Split the dataset into training and testing sets.
- Train Multinomial Naive Bayes and Logistic Regression classifiers.
- Compare the models using classification metrics.
- Create a reusable `predict_sentiment()` function.
- Test custom review sentences and display predicted sentiment.

**Algorithms and Techniques:**
- TF-IDF Vectorization
- Multinomial Naive Bayes
- Logistic Regression
- Text preprocessing

**Evaluation Metrics:**
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

**Dataset:** [Sample Sentiment Reviews Dataset](https://github.com/datasets/sample-reviews)

**Deliverable:** A Jupyter Notebook containing the text preprocessing pipeline, TF-IDF features, trained classifiers, evaluation results, and custom sentiment prediction function.

*Note: Verify that the linked sentiment dataset is accessible and contains the required review text and sentiment-label columns before starting Task 4.*

## 📁 Repository Structure

The repository can be organized as follows:

```text
machine-learning-data-science-internship/
│
├── README.md
├── requirements.txt
│
├── Task-1-Classification/
│   └── classification_pipeline.ipynb
│
├── Task-2-Housing-Regression/
│   └── housing_price_prediction.ipynb
│
├── Task-3-Customer-Churn/
│   └── customer_churn_prediction.ipynb
│
└── Task-4-Sentiment-Analysis/
    └── sentiment_classification.ipynb
```

Include only the tasks you have actually completed. Add generated charts or saved models in separate folders if they are part of your deliverables.

## ⚙️ Installation and Setup

### 1. Clone the Repository

Replace `YOUR_USERNAME` with your GitHub username.

```bash
git clone https://github.com/YOUR_USERNAME/machine-learning-data-science-internship.git
```

### 2. Navigate to the Project Directory

```bash
cd machine-learning-data-science-internship
```

### 3. Create a Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the notebook for the task you want to execute and run the cells in sequence.

## 📦 Dependencies

Create a `requirements.txt` file containing the following packages:

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
notebook
imbalanced-learn
xgboost
nltk
```

Install only the additional libraries needed for your selected tasks. Record tested package versions if reproducibility is required.

## 📊 Model Evaluation

The tasks use different evaluation metrics depending on the prediction problem.

| Metric | Application | Purpose |
|---|---|---|
| Accuracy | Classification | Measures the proportion of correct predictions |
| Precision | Classification | Measures how many positive predictions are correct |
| Recall | Classification | Measures how many actual positive cases are identified |
| F1-score | Classification | Balances precision and recall |
| Confusion Matrix | Classification | Shows correct and incorrect predictions by class |
| ROC-AUC | Binary Classification | Measures the model's ability to distinguish between classes |
| RMSE | Regression | Penalizes larger prediction errors |
| MAE | Regression | Measures average absolute prediction error |
| R² Score | Regression | Measures the proportion of target variance explained by the model |

Actual model results should be recorded after running the notebooks. Performance depends on data preparation, train/test splits, model settings, and validation methodology.

## 🔍 Best Practices

- Use a fixed random seed when reproducibility is required.
- Fit preprocessing steps only on the training data.
- Use pipelines to prevent data leakage.
- Apply SMOTE only to training data, never to the test set.
- Use cross-validation for reliable model selection.
- Tune hyperparameters using training data only.
- Report test-set results honestly without inventing performance values.
- Label visualizations and explain important findings.
- Keep notebooks organized, readable, and easy to execute.
- Do not upload passwords, API keys, private credentials, or unnecessary personal information.

## 🚀 Learning Outcomes

Through these tasks, I aim to strengthen my practical understanding of:

- Exploratory Data Analysis and data preprocessing.
- Classification and regression algorithms.
- Feature engineering and feature scaling.
- Regularization and hyperparameter tuning.
- Imbalanced classification and ensemble learning.
- Natural Language Processing and TF-IDF.
- Model evaluation and visualization.
- Reproducible machine learning workflows.
- GitHub-based project documentation and presentation.


## 👨‍💻 Author

**Tumu Indra Reddy**

B.Tech Student | Artificial Intelligence & Machine Learning | Data Science

- **GitHub:** [github.com/indratumu296-web](https://github.com/indratumu296-web)
- **LinkedIn:** [linkedin.com/in/tumu-indra-reddy](https://linkedin.com/in/tumu-indra-reddy)

## 📄 Disclaimer

This repository is intended for educational purposes and practical machine learning skill development as part of the internship task pack. Dataset sources and any third-party materials remain subject to their respective licenses and terms.

---

**Machine Learning & Data Science Internship | Datheon**

*Learn. Build. Evaluate. Document.*

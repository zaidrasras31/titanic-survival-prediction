# Titanic Survival Prediction

## Project Overview

A supervised machine learning project that predicts whether a passenger survived the Titanic disaster using passenger information such as age, sex, ticket class, and fare.

This project demonstrates data exploration, preprocessing, classification, model evaluation, and hyperparameter tuning using Python and Scikit-learn.

## Objective

Build and evaluate a machine learning classification model capable of predicting Titanic passenger survival.

The main objectives are:

- Explore and understand the Titanic dataset.
- Handle missing values and categorical variables.
- Apply feature engineering techniques.
- Train classification models.
- Evaluate model performance using appropriate metrics.
- Explore hyperparameter optimization using GridSearchCV.

## Problem Statement

The Titanic disaster resulted in different survival outcomes among passengers.

The goal is to predict the target variable `Survived` based on passenger characteristics.

This is a **binary classification problem**:

- `0` — Did not survive
- `1` — Survived

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computing |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning and evaluation |
| XGBoost | Additional classification algorithm |
| Jupyter Notebook | Interactive experimentation |
| Pytest | Automated testing |
| Git & GitHub | Version control |

## Dataset

**Source:** [Titanic — Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)

The dataset consists of:

- `train.csv` — Passenger information with survival labels.
- `test.csv` — Passenger information without survival labels.

Features include `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`, and `Cabin`.

## Project Structure

```text
titanic-survival-prediction/
├── .gitignore
├── LICENSE
├── Makefile
├── README.md
├── requirements.txt
├── main.py
├── titanic-survival.ipynb
│
├── data/
│   └── raw/
│       ├── train.csv
│       └── test.csv
│
├── src/
│   └── titanic/
│       ├── __init__.py
│       ├── preprocessing.py
│       ├── train.py
│       └── evaluate.py
│
└── tests/
    ├── __init__.py
    ├── test_preprocessing.py
    └── test_evaluate.py
```

**Note:** The `data/raw/` folder is created locally and excluded from Git.

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/zaidrasras31/titanic-survival-prediction.git
cd titanic-survival-prediction
```

### 2. Install Dependencies

Python 3.10 or later is recommended.

```bash
python -m pip install -r requirements.txt
```

Alternatively, if Make is installed:

```bash
make install
```

### 3. Download the Dataset

Download `train.csv` and `test.csv` from the Kaggle Titanic competition.

Place them inside:

```text
data/raw/
```

### 4. Train and Evaluate the Model

Run:

```bash
python main.py
```

This will:

1. Load the Titanic training dataset.
2. Create the cabin availability feature.
3. Split the dataset into training and validation sets.
4. Fit preprocessing on the training data.
5. Train a Random Forest classifier.
6. Display a classification report and confusion matrix.

### 5. Run the Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
titanic-survival.ipynb
```

The notebook contains the exploratory analysis and machine learning experiments.

### 6. Available Makefile Commands

| Command | Description |
|---|---|
| `make help` | Display available commands |
| `make install` | Install dependencies |
| `make notebook` | Launch Jupyter |
| `make test` | Run automated tests |
| `make clean` | Remove Python cache files |

## Machine Learning Workflow

### Exploratory Data Analysis

Explore passenger characteristics, missing values, feature distributions, and survival relationships.

### Data Preprocessing

Prepare numerical and categorical features using missing-value imputation and one-hot encoding.

### Feature Engineering

Create `Has_Cabin`, indicating whether a passenger has a recorded cabin number.

### Model Training

The reusable Python modules provide functions for Logistic Regression and Random Forest classification.

The notebook also explores additional machine learning experiments.

### Model Evaluation

Evaluate classification performance using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

### Hyperparameter Tuning

Use `GridSearchCV` with cross-validation to explore Random Forest hyperparameters, including:

- `n_estimators`
- `max_depth`
- `max_features`
- `min_samples_split`
- `min_samples_leaf`

## Tests

Automated tests are written using `pytest`.

Run:

```bash
python -m pytest tests/ -v
```

Or:

```bash
make test
```

The tests cover preprocessing behavior and classification evaluation metrics.

## GitHub Authentication and Push

For local development, configure Git and authenticate with GitHub using a supported method, such as Git Credential Manager or SSH.

After making changes:

```bash
git add .
git commit -m "Update Titanic survival prediction project"
git push origin main
```

These commands commit and push changes to the remote GitHub repository.

## Future Improvements

- Compare more classification algorithms.
- Explore additional feature engineering techniques.
- Improve hyperparameter optimization.
- Generate Kaggle submission predictions through the reusable Python pipeline.
- Add more automated tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

The Titanic dataset is provided separately by Kaggle and is subject to its own terms.

## Author

**Zaid Rasras**

Computer Engineering student at Birzeit University, interested in Artificial Intelligence, Machine Learning, and Data Science.

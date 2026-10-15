# Brain Stroke Prediction and Analysis using Machine Learning

A Django web application that analyses patient health data and compares nine machine learning algorithms for predicting the risk of a brain stroke. It has separate User and Admin modules, so patient records can be added, viewed and analysed through a simple web interface.

## Overview

Stroke is a leading cause of death and long-term disability, and early identification of high-risk people helps with timely prevention. This project uses the Kaggle Stroke Prediction dataset (age, hypertension, heart disease, average glucose level, BMI, smoking status, etc.) to train and compare classification models, and wraps the workflow in a web app.

## Dataset

- **Source:** Stroke Prediction Dataset (Kaggle), `healthcare-dataset-stroke-data.csv`
- **Size:** 5,110 records, 11 input features, 1 target (`stroke`)
- **Features:** gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status
- **Class balance:** only 249 stroke cases (about 4.9%), so the data is highly imbalanced

## Features

**User Module**
- Registration and login (accounts are activated by the admin)
- Add patient health data through a form
- View stored patient records (paginated)
- Run the machine learning comparison and see the accuracy of each algorithm

**Admin Module**
- Admin login
- View registered users and activate their accounts

## Machine Learning Workflow

1. **Data cleaning:** removed rows with missing BMI (4,909 records remain)
2. **Feature engineering:** grouped age, average glucose level and BMI into categories
3. **Encoding:** label encoding for binary columns, one-hot encoding for the rest
4. **Outlier handling:** capped extreme BMI values
5. **Scaling:** RobustScaler on age, BMI and glucose level
6. **Model comparison:** 5-fold cross-validation on nine algorithms: Logistic Regression, KNN, Random Forest, SVM, Decision Tree, Gradient Boosting, AdaBoost, XGBoost and LightGBM

## Tech Stack

- **Language:** Python
- **Framework:** Django
- **ML and data:** Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, Matplotlib, Seaborn
- **Frontend:** HTML, CSS, JavaScript
- **Database:** SQLite
- **Tools:** PyCharm

## Project Structure

```
Brainstroke/
├── Brainstroke/    # Django settings, main urls and views
├── users/          # user module (models, forms, views)
├── admins/         # admin module (views)
├── assets/         # templates and static files
├── media/          # dataset (CSV)
├── Document/       # project documentation
├── healthcare-dataset-stroke-data.csv
├── db.sqlite3
└── manage.py
```

## How to Run

```bash
# Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Install dependencies
pip install django django-pandas pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn

# Apply database migrations
python manage.py migrate

# Start the server
python manage.py runserver
```

Open `http://127.0.0.1:8000/` in your browser. Demo admin login: `admin` / `admin`.

## Future Improvements

- Handle class imbalance (SMOTE, class weights) and evaluate with precision, recall, F1 and ROC-AUC instead of accuracy alone
- Hyperparameter tuning of the best models
- Add a stroke prediction form that uses a trained stroke model
- Deploy the app online

## Disclaimer

This project is for educational purposes only and is not a substitute for professional medical diagnosis.

## Author

**Pavithra Mallampalli**
[LinkedIn](https://www.linkedin.com/in/pavithra-mallampalli-5a8697419) | mallampallip4@gmail.com

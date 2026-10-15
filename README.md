# Brain Stroke Prediction and Analysis using Machine Learning

A Django web application that analyses patient health data and compares nine machine learning algorithms for predicting the risk of a brain stroke. Early identification of high-risk patients helps with timely prevention and treatment.

**User Module:** register and log in (accounts are activated by the admin), add and view patient data, run the machine learning comparison and see the accuracy of each algorithm.  
**Admin Module:** log in, view registered users and activate their accounts.

## Dataset

- **Source:** Stroke Prediction Dataset (Kaggle), `healthcare-dataset-stroke-data.csv`
- **Size:** 5,110 records, 11 input features, 1 target (`stroke`)
- **Features:** gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status
- **Class balance:** only about 4.9% stroke cases, so the data is highly imbalanced

## Software Requirements

**Operating System:** Windows 10/11, Linux or macOS  
**Programming Language:** Python 3.8 or higher  
**Web Framework:** Django  
**Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, Matplotlib, Seaborn, django-pandas  
**Database:** SQLite  
**IDE / Editor:** PyCharm or VS Code  
**Web Browser:** Google Chrome or any modern browser

## Machine Learning Workflow

1. **Cleaning:** removed rows with missing BMI
2. **Feature engineering:** grouped age, glucose level and BMI into categories
3. **Encoding and scaling:** label and one-hot encoding, RobustScaler on numeric columns
4. **Model comparison:** 5-fold cross-validation on Logistic Regression, KNN, Random Forest, SVM, Decision Tree, Gradient Boosting, AdaBoost, XGBoost and LightGBM

**Tech stack:** Python, Django, Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, Matplotlib, Seaborn, SQLite

## How to Run

```bash
git clone https://github.com/PavithraMallampalli/Academic-Project.git
cd <repo-name>
pip install django django-pandas pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn
python manage.py migrate
python manage.py runserver
```

Open `http://127.0.0.1:8000/`. Demo admin login: `admin` / `admin`.

*For educational purposes only, not a substitute for professional medical diagnosis.*

**Author:** Pavithra Mallampalli | [LinkedIn](https://www.linkedin.com/in/pavithra-mallampalli-5a8697419) | mallampallip4@gmail.com

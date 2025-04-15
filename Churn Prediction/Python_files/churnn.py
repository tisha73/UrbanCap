import pandas as pd
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from pymongo import MongoClient
import joblib
import os
import optuna
from sklearn.ensemble import RandomForestClassifier,StackingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier

# Loading environment variables from .env file
load_dotenv()

mongodb_uri = os.getenv("MONGODB_URI")

client = MongoClient(mongodb_uri)
db = client["UrbanCap"]

churn_df = pd.DataFrame(list(db["churn"].find()))
churn_df.drop(columns={'_id','Booking_Frequency_Change','preferred_services','Spend_Change_Rate','Downgrade_Service_Usage'}, errors='ignore', inplace=True)

bookings_df = pd.DataFrame(list(db["Past_bookings"].find()))
bookings_df.drop(columns=['_id'], errors='ignore', inplace=True)

bookings_df['Booking_Date'] = pd.to_datetime(bookings_df['Booking_Date'])
data = pd.merge(churn_df, bookings_df, left_on='user_id', right_on='User_ID', how='left')
data['Service_Date'] = pd.to_datetime(data['Service_Date'])
data.sort_values(by=['User_ID', 'Service_Date'], inplace=True)

#  Feature Engineering

def calculate_spend_change_rate(df):
    n = len(df)
    if n < 4:
        return 1
    early = df.iloc[:n // 2]["Price"].mean()
    recent = df.iloc[n // 2:]["Price"].mean()
    return round(recent / early if early != 0 else 1, 2)

spend_change = data.groupby("User_ID").apply(calculate_spend_change_rate).reset_index(name="Spend_Change_Rate")

def booking_gap_change(df):
    if len(df) < 4:
        return 1
    df = df.sort_values("Service_Date")
    df["Gap"] = df["Service_Date"].diff().dt.days
    early = df["Gap"].iloc[1:len(df) // 2].mean()
    recent = df["Gap"].iloc[len(df) // 2:].mean()
    return round(early / recent if recent != 0 else 1, 2)

freq_change = data.groupby("User_ID").apply(booking_gap_change).reset_index(name="Booking_Frequency_Change")

def detect_downgrade_by_price(df):
    df = df.sort_values("Service_Date")
    n = len(df)
    if n < 4:
        return 0
    early = df.iloc[:n // 2]["Price"].mean()
    recent = df.iloc[n // 2:]["Price"].mean()
    return int(recent < early)

downgrade_price_flag = data.groupby("User_ID").apply(detect_downgrade_by_price).reset_index(name="Downgrade_By_Price")

def downgrade_by_service_id(df):
    df = df.sort_values("Service_Date")
    if len(df) < 4:
        return 0
    early_avg_id = df.iloc[:len(df) // 2]["Service_ID"].mean()
    recent_avg_id = df.iloc[len(df) // 2:]["Service_ID"].mean()
    return int(recent_avg_id < early_avg_id)

downgrade_id_flag = data.groupby("User_ID").apply(downgrade_by_service_id).reset_index(name="Downgrade_By_Service_ID")

combined_downgrade = downgrade_price_flag.copy()
combined_downgrade["Downgrade_Service_Usage"] = (
    downgrade_price_flag["Downgrade_By_Price"] | downgrade_id_flag["Downgrade_By_Service_ID"]
).astype(int)

user_features = data.groupby('User_ID').agg(
    Total_Bookings=('Booking_ID', 'count'),
    First_Booking_Date=('Booking_Date', 'min'),
    Last_Booking_Date=('Booking_Date', 'max'),
    Days_Since_Last_Booking=('Booking_Date', lambda x: (pd.Timestamp.today() - x.max()).days),
    Avg_Booking_Gap=('Booking_Date', lambda x: x.sort_values().diff().mean().days if x.count() > 1 else np.nan),
    Service_Types_Used=('Service_ID', lambda x: list(x.unique())),
    Total_Spend=('Price', 'sum'),
    Avg_Spend_Per_Booking=('Price', 'mean'),
    Last_Spend_Amount=('Price', lambda x: x.iloc[-1] if not x.empty else np.nan),
    Cancellation_Count=('Booking_Status', lambda x: (x == 'Cancelled').sum()),
).reset_index()

now = pd.Timestamp.today()
user_features['Account_Age_Days'] = (now - user_features['First_Booking_Date']).dt.days
user_features['Unique_Service_Types'] = user_features['Service_Types_Used'].apply(lambda x: len(x) if isinstance(x, list) else 0)

user_features.rename(columns={'User_ID': 'user_id'}, inplace=True)
spend_change.rename(columns={'User_ID': 'user_id'}, inplace=True)
freq_change.rename(columns={'User_ID': 'user_id'}, inplace=True)
combined_downgrade.rename(columns={'User_ID': 'user_id'}, inplace=True)

user_features = user_features \
    .merge(spend_change, on='user_id', how='left') \
    .merge(freq_change, on='user_id', how='left') \
    .merge(combined_downgrade[['user_id', 'Downgrade_Service_Usage']], on='user_id', how='left')

#  Preparing Features

engineered_features = [
    'Days_Since_Last_Booking', 'Avg_Booking_Gap', 'Total_Bookings',
    'Cancellation_Count', 'Total_Spend', 'Avg_Spend_Per_Booking', 'Last_Spend_Amount',
    'Unique_Service_Types','Spend_Change_Rate','Booking_Frequency_Change', 'Downgrade_Service_Usage'
]

additional_features = [
    'preferred_time_slots','booking_history_count','cancellation_rate','App_Logins_Per_Month',
    'Time_Spent_on_Platform (min)','Browsing_History_Depth','Search_Queries_Performed','Social_Media_Engagement',
    'subscription_status', 'average_rating_given', 'Complaints_Raised','Issue_Resolution_Time' ,
    'Click-Through-Rate (CTR)', 'Unsubscribed_Notifications'
]

X = user_features[engineered_features]
X[additional_features] = churn_df[additional_features]
churn_labels = churn_df[['user_id', 'Churn_Label']].drop_duplicates(subset='user_id')
y = user_features.merge(churn_labels, on='user_id', how='left')['Churn_Label']

#  Preprocessing pipeline

numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
 
def rf_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
    }
    rf = RandomForestClassifier(random_state=42, **params)
    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('classifier', rf)
    ])
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1', n_jobs=-1)
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(rf_objective, n_trials=30)

best_rf_params = study.best_params
best_rf = RandomForestClassifier(random_state=42, **best_rf_params)

#  Stacking Ensemble

base_learners = [
    ('lr', LogisticRegression(max_iter=1000)),
    ('xgb', XGBClassifier(
        n_estimators=150, learning_rate=0.1, use_label_encoder=False,
        eval_metric='logloss', random_state=42
    ))
]

stacking_model = StackingClassifier(
    estimators=base_learners,
    final_estimator=best_rf,
    cv=5,
    n_jobs=-1,
    passthrough=True
)

clf_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', stacking_model)
])

clf_pipeline.fit(X_train, y_train)

#  Evaluating

y_pred = clf_pipeline.predict(X_test)
y_probs = clf_pipeline.predict_proba(X_test)[:, 1]

print("\n Tuned Stacking Classifier Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_probs))

#  Saving artifacts

joblib.dump(preprocessor, "churn___preprocessor.pkl")
joblib.dump(clf_pipeline, "churn_stacking_model.pkl")

print("\n Preprocessor saved as churn___preprocessor.pkl")
print(" Full pipeline saved as churn_stacking_model.pkl")

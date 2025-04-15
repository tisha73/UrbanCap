import pandas as pd
import numpy as np
import joblib
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime

# Loading environment variables
load_dotenv()
mongodb_uri = os.getenv("MONGODB_URI")

# MongoDB connection
client = MongoClient(mongodb_uri)
db = client["UrbanCap"]

# Load saved pipeline
clf_pipeline = joblib.load("churn_stacking_model.pkl")

#  Feature Engineering 

def calculate_spend_change_rate(df):
    n = len(df)
    if n < 4:
        return 1
    early = df.iloc[:n // 2]["Price"].mean()
    recent = df.iloc[n // 2:]["Price"].mean()
    return round(recent / early if early != 0 else 1, 2)

def booking_gap_change(df):
    if len(df) < 4:
        return 1
    df = df.sort_values("Service_Date")
    df["Gap"] = df["Service_Date"].diff().dt.days
    early = df["Gap"].iloc[1:len(df) // 2].mean()
    recent = df["Gap"].iloc[len(df) // 2:].mean()
    return round(early / recent if recent != 0 else 1, 2)

def detect_downgrade_by_price(df):
    df = df.sort_values("Service_Date")
    n = len(df)
    if n < 4:
        return 0
    early = df.iloc[:n // 2]["Price"].mean()
    recent = df.iloc[n // 2:]["Price"].mean()
    return int(recent < early)

def downgrade_by_service_id(df):
    df = df.sort_values("Service_Date")
    if len(df) < 4:
        return 0
    early_avg_id = df.iloc[:len(df) // 2]["Service_ID"].mean()
    recent_avg_id = df.iloc[len(df) // 2:]["Service_ID"].mean()
    return int(recent_avg_id < early_avg_id)

#  Inference Function

def run_churn_prediction():
    churn_df = pd.DataFrame(list(db["churn"].find()))
    bookings_df = pd.DataFrame(list(db["Past_bookings"].find()))

    churn_df.drop(columns=['_id'], inplace=True, errors='ignore')
    bookings_df.drop(columns=['_id'], inplace=True, errors='ignore')

    bookings_df['Booking_Date'] = pd.to_datetime(bookings_df['Booking_Date'])
    bookings_df['Service_Date'] = pd.to_datetime(bookings_df['Service_Date'])

    data = pd.merge(churn_df, bookings_df, left_on='user_id', right_on='User_ID', how='left')
    data.sort_values(by=['User_ID', 'Service_Date'], inplace=True)

    spend_change = data.groupby("User_ID").apply(calculate_spend_change_rate).reset_index(name="Spend_Change_Rate")
    freq_change = data.groupby("User_ID").apply(booking_gap_change).reset_index(name="Booking_Frequency_Change")
    downgrade_price_flag = data.groupby("User_ID").apply(detect_downgrade_by_price).reset_index(name="Downgrade_By_Price")
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
        Cancellation_Count=('Booking_Status', lambda x: (x == 'Cancelled').sum())
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

    # Merging additional features
    additional_features = [
        'preferred_time_slots','booking_history_count','cancellation_rate','App_Logins_Per_Month',
        'Time_Spent_on_Platform (min)','Browsing_History_Depth','Search_Queries_Performed','Social_Media_Engagement',
        'subscription_status','average_rating_given','Complaints_Raised','Issue_Resolution_Time',
        'Click-Through-Rate (CTR)','Unsubscribed_Notifications'
    ]

    user_features[additional_features] = churn_df[additional_features]

    features_for_prediction = [
        'Days_Since_Last_Booking', 'Avg_Booking_Gap', 'Total_Bookings',
        'Cancellation_Count', 'Total_Spend', 'Avg_Spend_Per_Booking', 'Last_Spend_Amount',
        'Unique_Service_Types', 'Spend_Change_Rate', 'Booking_Frequency_Change', 'Downgrade_Service_Usage'
    ] + additional_features

    X_final = user_features[features_for_prediction]

    # Predict
    probs = clf_pipeline.predict_proba(X_final)[:, 1]
    preds = clf_pipeline.predict(X_final)

    results = user_features[['user_id']].copy()
    results['Churn_Probability'] = probs
    results['Churn_Prediction'] = preds

    print("\n Inference Complete:")
    print(results.head(10))
    return results

if __name__ == "__main__":
    run_churn_prediction()

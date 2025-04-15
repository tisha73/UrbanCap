import csv
from pymongo import MongoClient
import pandas as pd
from dotenv import load_dotenv
import os 

# Loading environment variables from .env file
load_dotenv()

mongodb_uri = os.getenv("MONGODB_URI")
client = MongoClient(mongodb_uri)

db = client["UrbanCap"]
# collection = db["UrbanCap"]
collection = db['Past_bookings']

csv_file = "Churn Prediction\\Data\\Book.csv"


# csv_file = "userUrban.xlsx"


df = pd.read_csv(csv_file)
# df.to_csv("churn_finall.csv", index=False)  # Saves without index



# with open(df, mode="r", encoding="utf-8") as f:
#     reader = csv.DictReader(f)  # Convert CSV rows into dictionaries
#     data = list(reader)  # Convert iterator to a list of dictionaries

data = df.to_dict(orient="records")

# Inserting data into MongoDB
collection.insert_many(data)

print("Data imported successfully!")
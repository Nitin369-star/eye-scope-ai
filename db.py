from pymongo import MongoClient
import os

client = MongoClient(os.getenv("MONGODB_URI"))
db = client["eye_disease_db"]

def save_prediction(data):
    db.predictions.insert_one(data)

from pymongo import MongoClient
import pandas as pd
from app.config.settings import MONGO_URI, DB_NAME, COLLECTION_NAME

class MongoService:
    def __init__(self):
        self.client = MongoClient(MONGO_URI)
        self.collection = self.client[DB_NAME][COLLECTION_NAME]

    def insert_prediction(self, record: dict):
        self.collection.insert_one(record)

    def fetch_all(self):
        data = list(self.collection.find({}, {"_id": 0}))
        return pd.DataFrame(data)

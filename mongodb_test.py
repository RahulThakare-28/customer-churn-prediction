from pymongo import MongoClient


def main():

    client = MongoClient("mongodb://localhost:27017/")
    db = client["churn_db"]
    collection = db["predictions"]
    collection.insert_one({"customer_id": 123, "predicted_churn": 1})
    print(list(collection.find()))

if __name__ == "__main__":
    main()
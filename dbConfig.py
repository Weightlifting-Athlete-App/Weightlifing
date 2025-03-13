from pymongo import MongoClient
from urllib.parse import quote_plus

# MongoDB connection settings
username = "PowerAppDB"
password = "res1234ear@"
escaped_username = quote_plus(username)
escaped_password = quote_plus(password)

MONGO_URI = f"mongodb+srv://{escaped_username}:{escaped_password}@powerappdb.6gd9t.mongodb.net/?retryWrites=true&w=majority&appName=PowerAppDB"
DATABASE_NAME = "PowerAppDB"
COLLECTION_NAME = "User_Data"

def get_db_connection():
    """
    Establishes a connection to MongoDB and returns the database and collection.
    """
    try:
        client = MongoClient(MONGO_URI)
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        print("Connected to MongoDB successfully!")
        return db, collection
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None, None
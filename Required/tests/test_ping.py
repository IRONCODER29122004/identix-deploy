import os
from pymongo import MongoClient
from pymongo.errors import PyMongoError

uri = os.environ.get("MONGODB_URI")
if not uri:
    raise SystemExit("MONGODB_URI not set")

client = MongoClient(uri, serverSelectionTimeoutMS=5000)

try:
    print("Pinging MongoDB...")
    result = client.admin.command("ping")
    print("Ping OK:", result)
except PyMongoError as e:
    print("Ping FAILED:", e)
finally:
    client.close()

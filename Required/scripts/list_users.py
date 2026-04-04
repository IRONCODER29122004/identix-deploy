"""
List all users in the MongoDB database
"""
from mongodb_utils import get_db

def list_all_users():
    try:
        db = get_db()
        users_collection = db['users']
        
        # Get all users
        users = list(users_collection.find({}))
        
        print(f"\n{'='*60}")
        print(f"Database: facial_landmarks_db")
        print(f"Collection: users")
        print(f"Total Users: {len(users)}")
        print(f"{'='*60}\n")
        
        if len(users) == 0:
            print("No users found in the database.")
            print("\nPossible reasons:")
            print("1. No one has registered yet")
            print("2. Users were deleted")
            print("3. Wrong database/collection name")
        else:
            for i, user in enumerate(users, 1):
                print(f"User {i}:")
                print(f"  ID: {user.get('_id')}")
                print(f"  Name: {user.get('name', 'N/A')}")
                print(f"  Email: {user.get('email', 'N/A')}")
                print(f"  Password Hash: {user.get('password', 'N/A')[:20]}... (truncated)")
                print(f"  Created: {user.get('created_at', 'N/A')}")
                print()
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    list_all_users()

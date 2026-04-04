"""
Clear all users from the MongoDB database
"""
from mongodb_utils import get_db

def clear_all_users():
    try:
        db = get_db()
        users_collection = db['users']
        
        # Count before deletion
        count_before = users_collection.count_documents({})
        print(f"Found {count_before} users in the database.")
        
        if count_before == 0:
            print("No users to delete.")
            return
        
        # Confirm deletion
        response = input(f"Are you sure you want to delete all {count_before} users? (yes/no): ")
        
        if response.lower() in ['yes', 'y']:
            # Delete all users
            result = users_collection.delete_many({})
            print(f"✓ Successfully deleted {result.deleted_count} users.")
        else:
            print("Deletion cancelled.")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    clear_all_users()

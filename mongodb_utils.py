"""MongoDB Utility Module

Provides a singleton MongoClient connected to MongoDB Atlas using the
MONGODB_URI environment variable. Exposes helper functions to obtain
the application database and to test connectivity.

Database Name: facial_landmarks_db (as per project specification)

Functions:
    get_db() -> Database instance
    test_ping() -> Runs a ping command and prints success/failure

Error Handling:
    - Missing MONGODB_URI env var: raises RuntimeError
    - Invalid URI / DNS issues: ConfigurationError captured
    - Authentication failures: OperationFailure captured
    - Connection / timeout issues: ServerSelectionTimeoutError, ConnectionFailure

Security:
    - NEVER logs the actual MongoDB URI or password
    - Only high-level error messages are emitted
"""
from __future__ import annotations
import os
from typing import Optional

from pymongo import MongoClient
from pymongo.errors import (
    ConfigurationError,
    ServerSelectionTimeoutError,
    OperationFailure,
    ConnectionFailure,
)

# Load .env automatically if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Internal singleton client reference
_client: Optional[MongoClient] = None

DB_NAME = "facial_landmarks_db"

def _create_client() -> MongoClient:
    """Create and return a new MongoClient using environment configuration.

    Reads MONGODB_URI from environment and establishes a connection with a
    5 second server selection timeout. Executes an initial ping to force
    server selection early and validate credentials.

    Returns:
        MongoClient: Connected client instance

    Raises:
        RuntimeError: For missing env var or connection/auth issues
    """
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError(
            "MONGODB_URI environment variable is not set. Set it before starting the application."
        )
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        # Force server selection & auth check
        client.admin.command("ping")
        return client
    except ConfigurationError as e:
        raise RuntimeError(f"MongoDB configuration error: {e}") from e
    except OperationFailure as e:
        raise RuntimeError(f"MongoDB authentication failed: {e}") from e
    except ServerSelectionTimeoutError as e:
        raise RuntimeError(f"MongoDB server selection timeout: {e}") from e
    except ConnectionFailure as e:
        raise RuntimeError(f"MongoDB connection failure: {e}") from e
    except Exception as e:  # Catch-all
        raise RuntimeError(f"MongoDB unexpected error: {e}") from e


def get_client() -> MongoClient:
    """Return the singleton MongoClient instance, creating it if needed."""
    global _client
    if _client is None:
        _client = _create_client()
    return _client


def get_db():
    """Return the application database object (facial_landmarks_db)."""
    return get_client()[DB_NAME]


def test_ping():
    """Run a ping command against the database and print result.

    Returns:
        dict: Result of the ping command.
    Raises:
        RuntimeError: If connection not established.
    """
    db = get_db()
    try:
        result = db.command("ping")
        ok = result.get("ok", 0) == 1.0
        print(f"MongoDB ping ok: {ok}")
        return result
    except Exception as e:
        raise RuntimeError(f"MongoDB ping failed: {e}") from e


if __name__ == "__main__":
    # Allow manual testing
    try:
        test_ping()
    except Exception as exc:
        print(f"Ping failed: {exc}")

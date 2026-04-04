"""Interactive MongoDB Atlas setup helper.

This script guides you through creating a proper MongoDB connection URI
and stores it in the local .env file for the project. It will:
 1. Ask for username, raw password, cluster host and optional params.
 2. URL-encode the password safely.
 3. Build the full URI.
 4. Test connectivity using mongodb_utils (ping).
 5. Write/update .env with MONGODB_URI=... if successful.

NOTE: You still must create the Atlas cluster, database user, and whitelist
your IP in the Atlas web UI before running this.

Usage:
  python setup_mongodb.py
"""
from __future__ import annotations
import os
import sys
import urllib.parse
from pathlib import Path

ENV_PATH = Path(".env")

def prompt(label: str, required: bool = True, default: str | None = None) -> str:
    while True:
        raw = input(f"{label}{' ['+default+']' if default else ''}: ").strip()
        if not raw and default is not None:
            return default
        if raw or not required:
            return raw
        print("  Value required.")

def build_uri(username: str, password: str, cluster: str, params: str) -> str:
    encoded_pwd = urllib.parse.quote(password, safe="")
    # Ensure cluster host does not include protocol
    cluster = cluster.replace("mongodb+srv://", "").rstrip("/")
    # Basic suffix if user omitted params
    suffix = params or "retryWrites=true&w=majority"
    # Append leading '?' if missing
    if suffix and not suffix.startswith("?"):
        suffix = "?" + suffix
    return f"mongodb+srv://{username}:{encoded_pwd}@{cluster}/{suffix}".rstrip("/")

def write_env(uri: str) -> None:
    lines = []
    if ENV_PATH.exists():
        existing = ENV_PATH.read_text().splitlines()
        replaced = False
        for line in existing:
            if line.startswith("MONGODB_URI="):
                lines.append(f"MONGODB_URI=\"{uri}\"")
                replaced = True
            else:
                lines.append(line)
        if not replaced:
            lines.append(f"MONGODB_URI=\"{uri}\"")
    else:
        lines.append(f"MONGODB_URI=\"{uri}\"")
    ENV_PATH.write_text("\n".join(lines) + "\n")

def test_uri(uri: str) -> bool:
    # Temporarily set env var for test
    os.environ["MONGODB_URI"] = uri
    try:
        from mongodb_utils import test_ping
        result = test_ping()
        ok = result.get("ok") == 1.0
        print("Ping OK" if ok else "Ping not OK")
        return ok
    except Exception as e:
        print(f"Ping failed: {e}")
        return False

def main():
    print("\n=== MongoDB Atlas URI Setup ===")
    print("Follow the prompts. Press Ctrl+C to abort.\n")
    print("PRE-REQUISITES:")
    print("  1. Atlas cluster created (Free M0 is fine).")
    print("  2. Database user created (username + password).")
    print("  3. Your IP or 0.0.0.0/0 whitelisted in Network Access.\n")

    username = prompt("Atlas database username")
    password = prompt("Atlas database password (raw, will be encoded)")
    cluster = prompt("Cluster host (e.g. cluster0.xxxxxx.mongodb.net)")
    params = prompt("Extra URI parameters (optional, blank to use default)", required=False)

    uri = build_uri(username, password, cluster, params)
    safe_display = uri.replace(password, "***")  # Mask raw password occurrence

    print("\nConstructed URI (password masked):")
    print(f"  {safe_display}\n")

    confirm = prompt("Test this URI now? (y/n)", default="y").lower()
    if confirm.startswith("y"):
        print("Testing connectivity...")
        if test_uri(uri):
            write_env(uri)
            print(f"\n✓ Connection succeeded. Saved to {ENV_PATH.resolve()} as MONGODB_URI.")
            print("Start your app: python Facial_Landmark_Project\\web_app\\landmark_app.py")
            return 0
        else:
            print("\n✗ Connection test failed. URI NOT saved.")
            print("Check: user credentials, IP whitelist, cluster readiness.")
            return 1
    else:
        print("Skipped test. Not writing .env. Re-run when ready.")
        return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(1)

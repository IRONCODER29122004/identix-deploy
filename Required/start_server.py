#!/usr/bin/env python
"""Direct Flask app startup for deepfake detection API"""
import sys
import os

# Change to Required directory so imports work
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Import everything from flask_hybrid_api
from flask_hybrid_api import app

if __name__ == '__main__':
    print("=" * 70)
    print("HYBRID DEEPFAKE DETECTION API")
    print("=" * 70)
    print("\nServer Information:")
    print(f"  Port: 5000")
    print(f"  Debug: False")
    print(f"  Detectors: Loaded")
    print(f"\nAvailable Endpoints:")
    for rule in app.url_map.iter_rules():
        if not rule.rule.startswith('/static'):
            methods = ','.join(rule.methods - {'HEAD', 'OPTIONS'})
            print(f"  {rule.rule} [{methods}]")
    print("\nStarting server...")
    print("=" * 70)
    print()
    
    # Run the server
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        use_reloader=False,
        threaded=True
    )

#!/usr/bin/env python
"""Flask app wrapper with proper output buffering"""
import sys
import os
import logging

# Ensure unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

# Enable logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

print("[WRAPPER] Starting Flask app wrapper...", flush=True)

from flask_hybrid_api import app

if __name__ == '__main__':
    print("[WRAPPER] App imported, starting server...", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False,
        threaded=True
    )

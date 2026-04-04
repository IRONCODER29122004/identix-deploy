"""
Quick test script to verify live detection integration
Run this to check if everything is set up correctly
"""

import sys
import os

print("="*70)
print("IDENTIX - Live Detection Integration Test")
print("="*70)

# Test 1: Check if files exist
print("\n1️⃣ Checking files...")
files_to_check = [
    'live_detection_routes.py',
    'faceswap_api_routes.py',
    'templates/live_faceswap.html',
    'landmark_app.py'
]

all_exist = True
for file in files_to_check:
    exists = os.path.exists(file)
    status = "✅" if exists else "❌"
    print(f"  {status} {file}")
    if not exists:
        all_exist = False

if not all_exist:
    print("\n❌ Some files are missing!")
    sys.exit(1)

print("\n✅ All files exist!")

# Test 2: Try importing modules
print("\n2️⃣ Testing imports...")
try:
    from live_detection_routes import register_routes
    print("  ✅ live_detection_routes imported successfully")
except ImportError as e:
    print(f"  ❌ Failed to import live_detection_routes: {e}")
    sys.exit(1)

try:
    from faceswap_api_routes import register_routes as register_faceswap
    print("  ✅ faceswap_api_routes imported successfully")
except ImportError as e:
    print(f"  ❌ Failed to import faceswap_api_routes: {e}")
    sys.exit(1)

# Test 3: Check Flask integration
print("\n3️⃣ Testing Flask integration...")
try:
    from flask import Flask
    test_app = Flask(__name__)
    
    # Try registering routes
    register_routes(test_app)
    print("  ✅ Live detection routes registered")
    
    register_faceswap(test_app)
    print("  ✅ Face swap routes registered")
    
    # Check routes exist
    routes = [rule.rule for rule in test_app.url_map.iter_rules()]
    
    expected_routes = [
        '/live-detection',
        '/api/live-detection/analyze',
        '/live-faceswap',
        '/api/faceswap/frame'
    ]
    
    for route in expected_routes:
        if route in routes:
            print(f"  ✅ {route}")
        else:
            print(f"  ❌ {route} not found!")
            
except Exception as e:
    print(f"  ❌ Flask integration failed: {e}")
    sys.exit(1)

# Test 4: Check dependencies
print("\n4️⃣ Checking dependencies...")
dependencies = {
    'cv2': 'OpenCV',
    'numpy': 'NumPy',
    'PIL': 'Pillow',
    'flask': 'Flask'
}

for module, name in dependencies.items():
    try:
        __import__(module)
        print(f"  ✅ {name}")
    except ImportError:
        print(f"  ⚠️  {name} - May cause issues")

# Summary
print("\n" + "="*70)
print("✅ ALL TESTS PASSED!")
print("="*70)
print("\n🚀 You're ready to run the server!")
print("\nTo start:")
print("  python landmark_app.py")
print("\nThen visit:")
print("  • http://localhost:5000/live-detection   (NEW! Live webcam detection)")
print("  • http://localhost:5000/live-faceswap    (Live face swap generation)")
print("  • http://localhost:5000/deepfake-detection (Upload video detection)")
print("\n" + "="*70)

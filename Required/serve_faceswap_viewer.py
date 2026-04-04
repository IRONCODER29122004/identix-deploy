#!/usr/bin/env python3
"""
Simple HTTP server to view FaceSwap results
Run: python serve_faceswap_viewer.py
Then open: http://localhost:8000
"""

import http.server
import socketserver
import os
from pathlib import Path
import webbrowser

# Get the directory of this script
SCRIPT_DIR = Path(__file__).parent
HTML_FILE = SCRIPT_DIR / 'faceswap_results_viewer.html'

PORT = 8000

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/viewer':
            self.path = '/faceswap_results_viewer.html'
        return super().do_GET()

    def end_headers(self):
        # Add headers to prevent caching
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        super().end_headers()

if __name__ == '__main__':
    # Change to the script directory
    os.chdir(SCRIPT_DIR)
    
    print("=" * 70)
    print("🎬 FaceSwap Results Viewer Server")
    print("=" * 70)
    print(f"\n✓ Serving from: {SCRIPT_DIR}")
    print(f"✓ HTML file: {HTML_FILE}")
    print(f"\n📍 Open your browser at:")
    print(f"   http://localhost:{PORT}")
    print(f"   http://127.0.0.1:{PORT}")
    print("\n💡 Features:")
    print("   • Video upload and playback")
    print("   • Frame-by-frame comparison")
    print("   • Generation statistics dashboard")
    print("   • Customizable playback settings")
    print("\n⌨️  Press Ctrl+C to stop the server")
    print("=" * 70 + "\n")
    
    try:
        with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
            print(f"✓ Server started on port {PORT}")
            print("✓ Waiting for connections...\n")
            
            # Try to open browser automatically
            try:
                webbrowser.open(f'http://localhost:{PORT}')
                print("✓ Opened browser automatically\n")
            except:
                pass
            
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n✓ Server stopped.")
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"\n❌ Error: Port {PORT} is already in use.")
            print("Try:")
            print(f"   - Wait a moment and try again")
            print(f"   - Use a different port by modifying PORT variable")
            print(f"   - Check what's using port {PORT}")
        else:
            print(f"\n❌ Error: {e}")

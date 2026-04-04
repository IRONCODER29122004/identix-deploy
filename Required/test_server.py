from flask import Flask, jsonify
import sys

app = Flask(__name__)

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'status': 'ok', 'message': 'Test server working'})

@app.route('/', methods=['GET'])
def root():
    return jsonify({'message': 'Test Flask server running'})

if __name__ == '__main__':
    print('Starting test server on port 5001...')
    sys.stdout.flush()
    app.run(host='0.0.0.0', port=5001, debug=False)

# Force TensorFlow to avoid oneDNN/MKL fused kernels on CPU to improve compatibility
# This prevents the "could not create a primitive" conv_transpose oneDNN error on some
# Windows/CPU setups. It slows some ops slightly but avoids runtime aborts.
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory, abort
from pathlib import Path
from werkzeug.utils import secure_filename
import threading
import json
import time
import hashlib
import secrets

from mongodb_utils import get_db

UPLOAD_FOLDER = 'data/uploads'
PIPELINES_FRAMES_DIR = 'data/pipelines_frames'
PIPELINES_CROPS_DIR = 'data/pipelines_crops'
ALLOWED_EXT = set(['mp4', 'mov', 'avi', 'mkv'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PIPELINES_FRAMES_DIR, exist_ok=True)
os.makedirs(PIPELINES_CROPS_DIR, exist_ok=True)

# Initialize MongoDB auth collection if configured.
try:
    db = get_db()
    auths_collection = db['auths']
    auths_collection.create_index('email', unique=True)
except Exception as e:
    auths_collection = None
    print(f"MongoDB auth unavailable: {e}")

STATUS_FILE = 'data/pipeline_status.json'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


def write_status(video_name, status_obj):
    try:
        all_status = {}
        if os.path.exists(STATUS_FILE):
            with open(STATUS_FILE, 'r') as f:
                all_status = json.load(f)
        all_status[video_name] = status_obj
        with open(STATUS_FILE, 'w') as f:
            json.dump(all_status, f)
    except Exception as e:
        print('Failed to write status:', e)


def run_pipeline_background(video_path, video_name):
    import pipeline_runner
    write_status(video_name, {'state': 'processing', 'started_at': time.time()})
    try:
        # run inference as part of pipeline if models exist
        res = pipeline_runner.process_video(video_path, fps=1, frames_out=PIPELINES_FRAMES_DIR, crops_out=PIPELINES_CROPS_DIR, video_name=video_name, run_inference=True)
        write_status(video_name, {'state': 'done', 'result': res, 'finished_at': time.time()})
    except Exception as e:
        write_status(video_name, {'state': 'error', 'error': str(e), 'finished_at': time.time()})


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return redirect(url_for('index'))
    f = request.files['video']
    if f.filename == '':
        return redirect(url_for('index'))
    if f and allowed_file(f.filename):
        filename = secure_filename(f.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(path)
        # start background thread to process the video
        t = threading.Thread(target=run_pipeline_background, args=(path, filename), daemon=True)
        t.start()
        return redirect(url_for('results', filename=filename))
    return redirect(url_for('index'))


@app.route('/results/<filename>')
def results(filename):
    crops_dir = PIPELINES_CROPS_DIR
    crops = []
    if os.path.exists(crops_dir):
        crops = [os.path.join(crops_dir, f) for f in os.listdir(crops_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    # read status
    status = {}
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, 'r') as f:
                status = json.load(f).get(filename, {})
        except Exception:
            status = {}
    return render_template('results.html', video=filename, crops=crops[:100], status=status)


@app.route('/status/<filename>')
def status_api(filename):
    # Return JSON status + list of crops and overlays for this filename
    status = {}
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, 'r') as f:
                status = json.load(f).get(filename, {})
        except Exception:
            status = {}
    crops = []
    crops_dir = PIPELINES_CROPS_DIR
    if os.path.exists(crops_dir):
        crops = sorted([os.path.join(crops_dir, f) for f in os.listdir(crops_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    # find overlays
    overlays = []
    overlays_by_model = {}
    if os.path.exists(crops_dir):
        for name in os.listdir(crops_dir):
            p = os.path.join(crops_dir, name)
            if os.path.isdir(p) and name.startswith('overlays_'):
                files = sorted([os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                overlays.extend(files)
                # derive model tag from folder name: overlays_{modelbasename}
                model_tag = name[len('overlays_'):]
                overlays_by_model[model_tag] = files
    return jsonify({'status': status, 'crops': crops, 'overlays': overlays, 'overlays_by_model': overlays_by_model})


@app.route('/inference/<filename>', methods=['POST'])
def inference_api(filename):
    # start background inference on crops for this upload
    video_name = filename
    def _run():
        try:
            write_status(video_name, {'state': 'inference_running'})
            import pipeline_runner
            # by default run all models; if request specified models, honor them
            # we stored requested models in the posted JSON body on the client; read it here
            models = None
            try:
                data = request.get_json(silent=True) or {}
                models = data.get('models')
            except Exception:
                models = None
            res = pipeline_runner.run_inference_on_crops(crops_dir=PIPELINES_CROPS_DIR, models=models)
            write_status(video_name, {'state': 'inference_done', 'result': res})
        except Exception as e:
            write_status(video_name, {'state': 'inference_error', 'error': str(e)})
    threading.Thread(target=_run, daemon=True).start()
    return jsonify({'started': True})


@app.route('/files/<path:filename>')
def serve_file(filename):
    # Serve files only from allowed directories to avoid accidental disclosure.
    # Allowed top-level directories (relative to project root):
    allowed = {'data', 'static'}
    root = os.getcwd()
    # Normalize and protect against path traversal
    requested = Path(filename)
    if not requested.parts:
        abort(404)
    top = requested.parts[0]
    if top not in allowed:
        # Not allowed to serve from other locations
        abort(403)
    # Build absolute path and verify it's inside the allowed directory
    allowed_dir = Path(root) / top
    resolved = (Path(root) / requested).resolve()
    try:
        allowed_resolved = allowed_dir.resolve()
    except Exception:
        abort(404)
    if not str(resolved).startswith(str(allowed_resolved)):
        abort(403)
    # Serve relative path under the allowed directory
    rel_path = Path(*requested.parts[1:]).as_posix() if len(requested.parts) > 1 else requested.name
    return send_from_directory(str(allowed_resolved), rel_path)


@app.route('/cleanup', methods=['POST'])
def cleanup():
    # expects form field 'which' in {'uploads','pipelines_crops'} and optional 'days'
    which = request.form.get('which', 'pipelines_crops')
    days = int(request.form.get('days', '30'))
    mapping = {'uploads': UPLOAD_FOLDER, 'pipelines_crops': PIPELINES_CROPS_DIR}
    target = mapping.get(which, PIPELINES_CROPS_DIR)
    try:
        import pipeline_runner
        removed = pipeline_runner.cleanup_old_files(target, older_than_days=days)
        return redirect(url_for('index'))
    except Exception as e:
        print('Cleanup failed:', e)
        return redirect(url_for('index'))


@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json(silent=True) or {}
        email = (data.get('email') or '').strip().lower()
        password = data.get('password') or ''
        name = (data.get('name') or 'User').strip() or 'User'

        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400
        if auths_collection is None:
            return jsonify({'error': 'Database not ready'}), 500

        hashed = hashlib.sha256(password.encode()).hexdigest()
        try:
            auths_collection.insert_one({'email': email, 'password': hashed, 'name': name})
        except Exception as e:
            if 'E11000' in str(e):
                return jsonify({'error': 'User already exists'}), 409
            raise

        from flask import session
        session['user_email'] = email
        session['user_name'] = name

        return jsonify({'success': True, 'message': 'Registration successful'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json(silent=True) or {}
        email = (data.get('email') or '').strip().lower()
        password = data.get('password') or ''
        remember = bool(data.get('remember'))

        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400
        if auths_collection is None:
            return jsonify({'error': 'Database not ready'}), 500

        hashed = hashlib.sha256(password.encode()).hexdigest()
        user = auths_collection.find_one({'email': email})
        if not user or user.get('password') != hashed:
            return jsonify({'error': 'Invalid credentials'}), 401

        from flask import session
        session['user_email'] = email
        session['user_name'] = user.get('name', 'User')
        session.permanent = remember
        return jsonify({'success': True, 'name': session['user_name']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/logout', methods=['POST'])
def logout():
    from flask import session
    session.clear()
    return jsonify({'success': True})


@app.route('/check-auth', methods=['GET'])
def check_auth():
    from flask import session
    if 'user_email' in session:
        return jsonify({'authenticated': True, 'email': session['user_email'], 'name': session.get('user_name', 'User')})
    return jsonify({'authenticated': False})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', '7860'))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

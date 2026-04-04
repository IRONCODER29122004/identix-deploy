import os
import json
from pathlib import Path

os.environ['AUTO_MODEL_EVAL_FRAMES'] = '8'
os.environ['AUTO_MODEL_KEYS'] = 'ffa_mpdv_v2,deepfakebench_meso4_v3,bisenet'

from landmark_app import app

ROOT = Path('.').resolve()
cases = [
    ('speaking_person.mp4', ROOT / 'faceswap_samples' / 'speaking_person.mp4', 'real'),
    ('test_person_video.mp4', ROOT / 'test_videos' / 'test_person_video.mp4', 'real'),
    ('output_deepfake.mp4', ROOT / 'output_deepfake.mp4', 'fake'),
]
models = ['bisenet', 'ffa_mpdv_v2', 'deepfakebench_meso4_v3', 'auto_best']

def norm(v):
    s = (v or '').lower()
    if 'authentic' in s or 'real' in s:
        return 'real'
    if 'deepfake' in s or 'fake' in s:
        return 'fake'
    if 'review' in s:
        return 'uncertain'
    return 'unknown'

rows = []
with app.test_client() as client:
    for name, path, truth in cases:
        for model in models:
            with open(path, 'rb') as f:
                resp = client.post('/detect_deepfake', data={'video': (f, path.name), 'max_frames': '10', 'model': model}, content_type='multipart/form-data')
            body = resp.get_json(silent=True) or {}
            rep = body.get('report', {})
            pred = norm(rep.get('verdict'))
            if pred in {'real','fake'}:
                outcome = 'OK' if pred == truth else ('FP' if truth == 'real' else 'FN')
            else:
                outcome = pred
            rows.append({
                'video': name,
                'truth': truth,
                'model': model,
                'status': resp.status_code,
                'prediction': pred,
                'verdict': rep.get('verdict'),
                'confidence': rep.get('confidence'),
                'auto_selected': ((rep.get('details') or {}).get('auto_selection') or {}).get('selected_model_key'),
                'outcome': outcome,
                'error': body.get('error')
            })

out = ROOT / 'benchmark_api_post_fix.json'
out.write_text(json.dumps(rows, indent=2), encoding='utf-8')
print(json.dumps(rows, indent=2))
print(f'SAVED:{out}')

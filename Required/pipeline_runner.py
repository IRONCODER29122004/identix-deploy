import os
import json
from pathlib import Path
import cv2
from tqdm import tqdm
import mediapipe as mp
import numpy as np

mp_face = mp.solutions.face_mesh
import json

STATUS_FILE = 'pipeline_status.json'


def write_status(video_name, status_obj):
    """Write a JSON status entry for video_name to STATUS_FILE."""
    try:
        all_status = {}
        if os.path.exists(STATUS_FILE):
            with open(STATUS_FILE, 'r') as f:
                all_status = json.load(f)
        all_status[video_name] = status_obj
        with open(STATUS_FILE, 'w') as f:
            json.dump(all_status, f)
    except Exception:
        pass


def extract_frames(video_path, out_dir='frames', fps=1, video_name=None):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video: {video_path}')
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, int(round(video_fps / fps)))
    saved = []
    i = 0; saved_i = 0
    pbar = tqdm(total=total_frames, desc='Extracting frames')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % frame_interval == 0:
            fname = os.path.join(out_dir, f'frame_{saved_i:05d}.jpg')
            cv2.imwrite(fname, frame)
            saved.append(fname)
            saved_i += 1
            if video_name:
                write_status(video_name, {'state': 'extracting_frames', 'extracted': saved_i, 'total_est': total_frames})
        i += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    return saved


def detect_landmarks_for_frames(frame_paths, max_frames=None, static_image_mode=True, refine_landmarks=False, video_name=None):
    results = {}
    total = len(frame_paths[:max_frames]) if max_frames else len(frame_paths)
    with mp_face.FaceMesh(static_image_mode=static_image_mode, max_num_faces=1, refine_landmarks=refine_landmarks) as face_mesh:
        for idx, p in enumerate(frame_paths[:max_frames]):
            img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            r = face_mesh.process(img)
            if r and r.multi_face_landmarks:
                lm = r.multi_face_landmarks[0]
                pts = [(pt.x, pt.y, pt.z) for pt in lm.landmark]
                results[p] = {'landmarks': pts}
            else:
                results[p] = {'landmarks': None}
            if video_name:
                write_status(video_name, {'state': 'landmarking', 'processed': idx + 1, 'total_frames': total})
    return results


def bbox_from_landmarks(landmarks, img_shape, pad=0.2):
    h, w = img_shape[:2]
    xy = __import__('numpy').array([[x * w, y * h] for (x, y, _) in landmarks])
    x_min, y_min = xy.min(axis=0); x_max, y_max = xy.max(axis=0)
    w_box = x_max - x_min; h_box = y_max - y_min
    x_min = max(0, int(x_min - pad * w_box))
    y_min = max(0, int(y_min - pad * h_box))
    x_max = min(w, int(x_max + pad * w_box))
    y_max = min(h, int(y_max + pad * h_box))
    return x_min, y_min, x_max, y_max


def pick_best_face_region_and_save(frames_list, landmarks_dict, out_dir='crops', min_landmarks=10, size=(256, 256)):
    os.makedirs(out_dir, exist_ok=True)
    metadata = []
    for p in frames_list:
        lm_info = landmarks_dict.get(p)
        img = cv2.imread(p)
        if lm_info is None or lm_info.get('landmarks') is None:
            h, w = img.shape[:2]
            cx, cy = w // 2, h // 2
            half = min(w, h) // 4
            crop = img[cy - half:cy + half, cx - half:cx + half] if half > 0 else img
            out_p = os.path.join(out_dir, os.path.basename(p))
            cv2.imwrite(out_p, cv2.resize(crop, size))
            metadata.append({'frame': p, 'crop': out_p, 'method': 'center'})
            continue
        lm = lm_info['landmarks']
        if len(lm) < min_landmarks:
            h, w = img.shape[:2]
            cx, cy = w // 2, h // 2
            half = min(w, h) // 4
            crop = img[cy - half:cy + half, cx - half:cx + half] if half > 0 else img
            out_p = os.path.join(out_dir, os.path.basename(p))
            cv2.imwrite(out_p, cv2.resize(crop, size))
            metadata.append({'frame': p, 'crop': out_p, 'method': 'fallback'})
            continue
        x_min, y_min, x_max, y_max = bbox_from_landmarks(lm, img.shape)
        crop = img[y_min:y_max, x_min:x_max]
        try:
            out_p = os.path.join(out_dir, os.path.basename(p))
            cv2.imwrite(out_p, cv2.resize(crop, size))
            metadata.append({'frame': p, 'crop': out_p, 'method': 'bbox', 'bbox': [x_min, y_min, x_max, y_max]})
        except Exception:
            h, w = img.shape[:2]
            cx, cy = w // 2, h // 2
            half = min(w, h) // 4
            crop = img[cy - half:cy + half, cx - half:cx + half] if half > 0 else img
            out_p = os.path.join(out_dir, os.path.basename(p))
            cv2.imwrite(out_p, cv2.resize(crop, size))
            metadata.append({'frame': p, 'crop': out_p, 'method': 'exception'})
        # optionally write incremental status to a running status file if present
        try:
            # detect if a status structure exists for this video by checking pipeline_status.json
            if os.path.exists(STATUS_FILE):
                # no video_name available here, but we can leave higher-level code to write final statuses
                pass
        except Exception:
            pass
    with open(os.path.join(out_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    return metadata


def process_video(video_path, fps=1, frames_out='pipelines_frames', crops_out='pipelines_crops', max_frames=None, video_name=None, run_inference=False):
    """Full pipeline: extract frames, landmark frames, pick best-face crops.

    If video_name is provided, write progress into STATUS_FILE under that key.
    """
    if video_name:
        write_status(video_name, {'state': 'extracting_frames'})
    frames = extract_frames(video_path, out_dir=frames_out, fps=fps)
    if max_frames:
        frames = frames[:max_frames]
    if video_name:
        write_status(video_name, {'state': 'landmarking', 'total_frames': len(frames)})
    lm = detect_landmarks_for_frames(frames, max_frames=max_frames)
    # after landmarking update counts
    detected = sum(1 for p in frames if lm.get(p) and lm[p].get('landmarks') is not None)
    if video_name:
        write_status(video_name, {'state': 'saving_crops', 'frames': len(frames), 'detected_faces': detected})
    metadata = pick_best_face_region_and_save(frames, lm, out_dir=crops_out)
    if video_name:
        write_status(video_name, {'state': 'done', 'frames': len(frames), 'metadata_count': len(metadata)})
    if run_inference:
        try:
            infer_res = run_inference_on_crops(crops_out)
            if video_name:
                write_status(video_name, {'state': 'inference_done', 'inference': infer_res})
        except Exception as e:
            if video_name:
                write_status(video_name, {'state': 'inference_error', 'error': str(e)})
    return {'frames': frames, 'metadata': metadata}


def find_model_candidate(model_type='unet'):
    """Return path to a model file if available for model_type. Supports 'unet','deeplab','vit'."""
    candidates = []
    if model_type == 'unet':
        candidates = ['models/unet_model.keras', 'models/unet_smoke.keras']
    elif model_type == 'deeplab':
        candidates = ['models/deeplab_model_final.keras', 'models/deeplab_model_stage1.keras', 'models/deeplab_model.keras']
    elif model_type == 'vit':
        candidates = ['models/vit_model.keras', 'models/vit_full.keras', 'models/vit_smoke.keras']
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def run_inference_on_crops(crops_dir='crops', output_dir=None, models=None):
    """Run available models on crops and save overlayed outputs under output_dir/{model_name}/

    Args:
      crops_dir: directory with crop image files
      output_dir: where to write overlays (defaults to crops_dir)
      models: optional list of model types to run (e.g., ['unet','deeplab']). If None, run all.
    """
    import tensorflow as tf
    os.makedirs(crops_dir, exist_ok=True)
    crop_files = sorted([os.path.join(crops_dir, f) for f in os.listdir(crops_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))])
    if not crop_files:
        return {'skipped': True, 'reason': 'no crops'}
    if output_dir is None:
        output_dir = crops_dir
    results = {}
    # Try each model type
    candidate_types = ['unet','deeplab','vit']
    if models:
        candidate_types = [m for m in candidate_types if m in models]
    for mtype in candidate_types:
        model_path = find_model_candidate(mtype)
        if not model_path:
            results[mtype] = {'available': False}
            continue
        try:
            model = tf.keras.models.load_model(model_path)
        except Exception as e:
            results[mtype] = {'available': False, 'error': str(e)}
            continue
        out_folder = os.path.join(output_dir, f'overlays_{os.path.splitext(os.path.basename(model_path))[0]}')
        os.makedirs(out_folder, exist_ok=True)
        count = 0
        try:
            for cf in crop_files:
                img = cv2.imread(cf)
                inp = cv2.resize(img, (256,256)) / 255.0
                # handle models that expect 4 channels
                if model.input_shape[-1] == 4:
                    h,w,_ = inp.shape
                    blank = np.zeros((h,w,1), dtype=inp.dtype)
                    inp = np.concatenate([inp, blank], axis=-1)
                pred = model.predict(np.expand_dims(inp, axis=0))
                pred_mask = np.argmax(pred, axis=-1)[0].astype(np.uint8)
                # colorize mask
                cmap = (np.array([[0,0,0]] + [[int((i*37)%255), int((i*61)%255), int((i*91)%255)] for i in range(1,256)])).astype(np.uint8)
                color = cmap[pred_mask]
                color = cv2.resize(color, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                overlay = cv2.addWeighted(img, 0.6, color, 0.4, 0)
                out_p = os.path.join(out_folder, os.path.basename(cf))
                cv2.imwrite(out_p, overlay)
                count += 1
        except Exception as e:
            results[mtype] = {'available': False, 'error': str(e)}
            # continue to next model
            continue
        results[mtype] = {'available': True, 'model': model_path, 'count': count, 'out_folder': out_folder}
    return results

def cleanup_old_files(directory, older_than_days=30):
    """Remove files in `directory` older than `older_than_days` days. Returns list of removed paths."""
    import time
    removed = []
    if not os.path.exists(directory):
        return removed
    cutoff = time.time() - older_than_days * 24 * 3600
    for fname in os.listdir(directory):
        p = os.path.join(directory, fname)
        try:
            if os.path.isfile(p) and os.path.getmtime(p) < cutoff:
                os.remove(p)
                removed.append(p)
            # optionally remove empty dirs
            if os.path.isdir(p) and not os.listdir(p):
                os.rmdir(p)
        except Exception:
            continue
    return removed

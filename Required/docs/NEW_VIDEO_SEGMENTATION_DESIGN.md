# NEW Video Segmentation Design - From Scratch
**Date**: 2025-11-26  
**Goal**: Simple, reliable multi-person video segmentation that actually works

---

## Problem with Current Implementation
1. Over-complicated tracking with multiple scoring criteria
2. Unclear which person is "main" - weighted scoring not working
3. Frame diversity logic too complex
4. Too many parameters to tune

## New Simple Design

### Core Principle: KISS (Keep It Simple, Stupid)
**Main person = Person who appears in MOST frames AND has LARGEST average face size**

---

## Algorithm (5 Clear Steps)

### Step 1: Extract Frames
- Use existing `extract_frames()` function
- Get uniformly sampled frames from video (max 100)

### Step 2: Detect Faces Per Frame
- Use Haar Cascade face detection (existing `detect_faces()`)
- Keep parameters moderate: scaleFactor=1.1, minNeighbors=5, minSize=(80,80)
- Simple validation: check aspect ratio, not too small, not at edges

### Step 3: Track Persons Across Frames
**Simple IoU-based tracking:**
```
For each detected face in current frame:
    Find best match in previous frame using IoU (Intersection over Union)
    If IoU > 0.3:
        Same person - keep same ID
    Else:
        New person - assign new ID
```

**Data to track per person:**
- List of frame numbers where they appear
- List of bounding boxes (one per appearance)
- List of face sizes (width * height)

### Step 4: Identify Main Person
**Simple 2-criteria ranking:**
```python
For each tracked person:
    screen_time = number of frames they appear in
    avg_size = average of their face sizes
    
    # Simple product (both must be high)
    score = screen_time * avg_size

Main person = person with highest score
```

**Why this works:**
- Background person: Small face OR appears in few frames → LOW score
- Main person: Large face AND appears in many frames → HIGH score
- No complex weighting needed - multiplication naturally handles it

### Step 5: Select Best Frame Per Landmark
**For main person only:**

1. **Get all their frames** with landmark predictions
2. **For each landmark (1-10):**
   - Calculate quality score for this landmark in each frame
   - Sort frames by quality
   - Pick TOP frame that hasn't been used yet
   - If all used, pick best regardless

**Quality score per landmark (simple):**
```python
# Percentage of pixels belonging to this landmark
landmark_pixels = (prediction == landmark_id).sum()
total_face_pixels = prediction.size
quality = (landmark_pixels / total_face_pixels) * 100
```

**Diversity guarantee:**
- Track which frame numbers are already used
- Prefer unused frames (natural diversity)
- No complex priority ordering needed

---

## Implementation Structure

```python
def process_video_landmarks_v2(video_path, max_frames=100):
    """
    Simple, reliable video segmentation
    """
    # Step 1: Extract frames
    frames, frame_numbers, fps, total = extract_frames(video_path, max_frames)
    
    # Step 2 & 3: Detect and track faces
    persons = {}  # person_id -> {'frames': [], 'bboxes': [], 'sizes': []}
    last_frame_faces = {}  # id -> bbox (for IoU matching)
    
    for frame_idx, frame in enumerate(frames):
        detected = detect_faces(frame)
        current_matches = {}
        
        for bbox in detected:
            # Match to existing person via IoU
            person_id = match_face(bbox, last_frame_faces)
            
            # Store data
            persons[person_id]['frames'].append(frame_numbers[frame_idx])
            persons[person_id]['bboxes'].append(bbox)
            persons[person_id]['sizes'].append(bbox[2] * bbox[3])
            
            current_matches[person_id] = bbox
        
        last_frame_faces = current_matches
    
    # Step 4: Find main person
    main_id = None
    best_score = 0
    
    for pid, data in persons.items():
        screen_time = len(data['frames'])
        avg_size = np.mean(data['sizes'])
        score = screen_time * avg_size
        
        if score > best_score:
            best_score = score
            main_id = pid
    
    # Step 5: Select best frames per landmark for main person
    main_data = persons[main_id]
    best_frames = {}
    used_frame_nums = set()
    
    for landmark_id in range(1, 11):
        # Predict landmarks for all frames of main person
        candidates = []
        for i, frame_num in enumerate(main_data['frames']):
            frame_idx = frame_numbers.index(frame_num)
            frame = frames[frame_idx]
            bbox = main_data['bboxes'][i]
            
            prediction, colored, crop = predict_landmarks_for_face(frame, bbox)
            quality = calculate_simple_quality(prediction, landmark_id)
            
            candidates.append({
                'frame_num': frame_num,
                'quality': quality,
                'prediction': prediction,
                'colored': colored,
                'crop': crop,
                'image': frame
            })
        
        # Sort by quality, prefer unused frames
        candidates.sort(key=lambda x: (
            x['frame_num'] not in used_frame_nums,  # Unused first
            x['quality']  # Then by quality
        ), reverse=True)
        
        best = candidates[0]
        best_frames[landmark_id] = best
        used_frame_nums.add(best['frame_num'])
    
    return main_id, persons, best_frames
```

---

## Key Simplifications

| Old (Complex) | New (Simple) |
|---------------|--------------|
| IoU + Size + Center scoring | Simple IoU matching (threshold 0.3) |
| 40/30/30 weighted scoring | screen_time × avg_size (product) |
| Landmark priority order | Natural diversity via unused frames |
| Size history running average | Just track all sizes, use mean |
| Complex validation (skin, eyes, variance) | Basic validation (aspect, size, position) |

---

## Expected Behavior

**Input**: Video with 2 people (Person A close to camera, Person B in background)

**Step 2-3 Detection:**
- Person A detected in 80/100 frames (avg size: 10,000 px²)
- Person B detected in 30/100 frames (avg size: 3,000 px²)

**Step 4 Scoring:**
- Person A score: 80 × 10,000 = 800,000 ✓ MAIN
- Person B score: 30 × 3,000 = 90,000

**Step 5 Frame Selection:**
- Landmark 1: Frame 45 (quality 85%, unused)
- Landmark 2: Frame 12 (quality 90%, unused)
- Landmark 3: Frame 78 (quality 88%, unused)
- ... (6-8 unique frames naturally)

---

## Testing Checklist

- [ ] Solo video (1 person) → Correctly identified as main
- [ ] Duo video (main + background) → Larger person identified as main
- [ ] Frame diversity → 6-8 unique frames used
- [ ] Quality scores → Realistic (20-95% range)
- [ ] Console output → Clear logging of detection and selection

---

## What to LOG

```
Processing 100 frames from video...

Detected persons:
- Person 0: 80 frames, avg size 10,000 px² → Score: 800,000
- Person 1: 30 frames, avg size 3,000 px² → Score: 90,000

✓ MAIN PERSON: Person 0 (appeared in 80/100 frames, 80% screen time)

Selecting best frames per landmark...
- Landmark 1 (Skin): Frame 45, quality 85%
- Landmark 2 (L Eyebrow): Frame 12, quality 90%
...
✓ Selected 8 unique frames

Processing complete!
```

---

**Next**: Implement this design, replacing current process_video_landmarks

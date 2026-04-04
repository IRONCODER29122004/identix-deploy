# 🎬 FaceSwap Results Viewer - User Guide

## Overview

A **professional HTML dashboard** to view, compare, and manage your FaceSwap generation results with interactive features and statistics tracking.

## Features

### 📹 **Video Viewer Tab**
- **Upload or drag-drop** deepfake videos
- **Full playback controls** (play, pause, seek, volume, fullscreen)
- **Recent results** list showing all processed videos
- **Download videos** for sharing or further analysis

### 🔀 **Frame Comparison Tab**
- **Side-by-side comparison** of source and result frames
- **Upload source face** and generated frame images
- **Visual quality assessment** of face swap
- **Download comparison** View

### 📊 **Statistics Tab**
- **Real-time metrics** dashboard:
  - Frames processed
  - Success rate (%)
  - Processing time
  - Video duration
- **Detailed parameters** (model, blending method, landmarks)
- **JSON stats import** - paste generation stats for detailed tracking
- **Performance analysis**

### ⚙️ **Settings Tab**
- **Video playback**: Autoplay, loop, playback speed (0.5x - 2x)
- **Export settings**: Format (MP4, WebM, GIF), quality (720p, 1080p, 4K)
- **Theme customization**: Color schemes (Purple, Blue, Green, Dark)
- **Auto-save** to browser local storage

---

## How to Use

### **Option 1: Run the Server** (Recommended)

```bash
cd d:\link2\Capstone\ 4-1\Code_try_1\Required\
python serve_faceswap_viewer.py
```

Then open your browser to:
```
http://localhost:8000
```

### **Option 2: Open Directly**

Simply double-click:
```
faceswap_results_viewer.html
```

(Opens in your default browser)

---

## Quick Start

### **View a Generated Deepfake**

1. Click the **📹 Video Viewer** tab
2. Drag & drop or click to upload your `output_deepfake.mp4`
3. Use playback controls to review
4. Click **⬇️ Download Video** to save

### **Compare Source & Result Frames**

1. Click **🔀 Frame Comparison** tab
2. Upload **source face image** (original face)
3. Upload **result frame** (frame from generated video)
4. Visual side-by-side comparison displays
5. Button to download comparison for documentation

### **View Generation Statistics**

1. Click **📊 Statistics** tab
2. See **auto-populated stats**:
   - Frames processed: Shows total frames
   - Success rate: % of successful swaps
   - Processing time: How long generation took
   - Duration: Generated video length

3. **Paste JSON stats** in the text area for detailed tracking:
```json
{
  "frames_processed": 150,
  "frames_failed": 5,
  "fps": 30,
  "total_frames": 155,
  "processing_time_seconds": 450,
  "success_rate": 96.77,
  "source_face_quality": "High",
  "target_video_clarity": "Good"
}
```

4. Click **📈 Load Stats** to display metrics

### **Configure Playback**

1. Click **⚙️ Settings** tab
2. Enable **Autoplay** or **Loop** videos
3. Set **Playback speed** (0.5x to 2x)
4. Choose **export format** (MP4, WebM, GIF)
5. Select **quality** (720p, 1080p, 4K)
6. Click **💾 Save Settings** - persisted to browser

---

## Integration with Python

After generating a deepfake in Jupyter, save stats and open the viewer:

```python
# In your notebook after swap_faces_in_video()
result = {
    "frames_processed": 150,
    "frames_failed": 5,
    "fps": 30,
    "total_frames": 155,
    "processing_time_seconds": 450,
    "success_rate": ((150 / 155) * 100),
    "source_face_quality": "High",
    "target_video_clarity": "Good"
}

# Save stats to JSON
import json
with open('faceswap_stats.json', 'w') as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
```

Then:
1. Copy the JSON output
2. Open the viewer → **Statistics** tab
3. Paste into JSON textarea
4. Click **📈 Load Stats**

---

## Browser Compatibility

✅ **Supported:**
- Chrome/Chromium (recommended)
- Firefox
- Edge
- Safari
- Any modern browser with HTML5 video support

✅ **Features:**
- Responsive design (desktop, tablet, mobile)
- Works offline after initial load
- Local file upload (no cloud uploads)
- Browser storage for settings persistence

---

## File Formats

### **Video**
- MP4 (H.264)
- WebM (VP9)
- OGG/Theora
- MOV
- AVI (most browsers)

### **Images**
- JPG/JPEG
- PNG
- WebP
- BMP

### **Statistics**
- JSON (plain text)

---

## Tips & Tricks

### 💡 **For Best Results**

1. **Video Quality**: Use high-quality source and target videos
2. **Frame Rate**: 24-30 FPS recommended for smooth playback
3. **Resolution**: 720p - 1080p sweet spot (4K needs more processing)
4. **Source Face**: Clear, frontal face in good lighting

### 📊 **Statistics Tracking**

Keep a JSON file with generation parameters:
```json
{
  "source_face": "celebrity_A.jpg",
  "target_video": "interview_B.mp4",
  "model_version": "MediaPipe v1.0",
  "timestamp": "2026-03-09T14:30:00Z",
  "frames_processed": 450,
  "success_rate": 95.5,
  "generation_time_minutes": 12,
  "output_file": "deepfake_AB.mp4",
  "notes": "High quality swap, good face alignment"
}
```

### 🎨 **Theme Customization**

Choose your favorite color scheme:
- **Purple** (default, calming)
- **Blue** (professional)
- **Green** (sleek)
- **Dark** (eye-friendly, night mode)

---

## Troubleshooting

### Video won't play
- ✓ Check video format (use MP4 for best compatibility)
- ✓ Try converting with FFmpeg:
  ```bash
  ffmpeg -i input.avi -c:v libx264 -crf 23 output.mp4
  ```

### Server won't start
- ✓ Port 8000 in use? Kill existing process or use different port
- ✓ Edit `serve_faceswap_viewer.py`, change `PORT = 8000` to `PORT = 8080`

### Images not loading
- ✓ Ensure JPG/PNG files are not corrupted
- ✓ Try saving again with better quality

### Settings not saving
- ✓ Check browser cookies/storage are enabled
- ✓ Not in private/incognito mode (storage disabled)

---

## File Structure

```
Required/
├── faceswap_results_viewer.html      ← Main viewer (open this!)
├── serve_faceswap_viewer.py          ← Local server script
├── faceswap_tech_and_detection_starter.ipynb
├── test_videos/
│   ├── real_sample_5s_a.mp4
│   ├── real_sample_10s_b.mp4
│   └── real_sample_640x360_c.mp4
└── output_deepfake.mp4               ← Your generated deepfakes
```

---

## Technical Details

### **Frontend Stack**
- Vanilla HTML5 / CSS3 / JavaScript (no dependencies!)
- Native HTML5 `<video>` element
- IndexedDB / LocalStorage for persistence
- Responsive Grid/Flexbox layout

### **Browser APIs Used**
- FileReader API (local file upload)
- Canvas API (image processing)
- Video API (playback)
- localStorage (settings persistence)

### **Performance**
- Zero external CDN dependencies
- All CSS/JS inline (single file)
- Optimized for mobile/tablet
- Fast load times

---

## FAQ

**Q: Is my data private?**  
A: Yes! All processing happens locally in your browser. No data is sent to any server.

**Q: Can I use this offline?**  
A: Yes, after opening once, it works completely offline.

**Q: What's the max video size?**  
A: Browser limit (usually 2GB+). Depends on available RAM.

**Q: Can I export statistics?**  
A: Yes, copy from JSON textarea or download as text file.

**Q: Does this work on mobile?**  
A: Yes! Fully responsive design for phones/tablets.

---

## Support

For issues or feature requests:
1. Check browser console (F12 → Console tab)
2. Verify HTML file is accessible
3. Try different browser
4. Ensure JavaScript is enabled

---

**Made with ❤️ for FaceSwap forensics and deepfake quality assurance**

Version: 1.0 | Last Updated: March 2026

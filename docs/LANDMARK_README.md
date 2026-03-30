# Facial Landmark Generation Project

A complete web application for generating facial landmarks using deep learning (BiSeNet model). This project includes training pipeline and a localhost web interface for real-time landmark detection.

## 🌟 Features

- **BiSeNet Architecture**: State-of-the-art bilateral segmentation network
- **11 Landmark Classes**: Detects multiple facial features
- **Web Interface**: User-friendly drag-and-drop interface
- **Real-time Processing**: Fast inference on uploaded images
- **Visualization**: Multiple views - original, prediction mask, and overlay
- **Statistics**: Detailed landmark pixel counts

## 📁 Project Structure

```
Code_try_1/
├── new.ipynb                    # Training notebook
├── landmark_app.py              # Flask web application
├── templates/
│   └── landmark_index.html      # Frontend interface
├── train/
│   ├── images/                  # Training images
│   └── labels/                  # Training labels
├── val/
│   ├── images/                  # Validation images
│   └── labels/                  # Validation labels
├── test/
│   ├── images/                  # Test images
│   └── labels/                  # Test labels
└── best_model.pth               # Trained model weights (after training)
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster training)

### Install Dependencies

```powershell
# Install PyTorch (CPU version)
pip install torch torchvision torchaudio

# Or for GPU version (check pytorch.org for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install flask pillow numpy matplotlib opencv-python tqdm scikit-learn
```

### (Optional) MongoDB Atlas Integration

If you want to persist uploads or user data, set up MongoDB Atlas and create a `.env` with `MONGODB_URI`.

Quick setup using the interactive helper:
```powershell
python setup_mongodb.py
```
This will:
- Prompt for your Atlas username, password, and cluster host (e.g. `cluster0.xxxxx.mongodb.net`)
- URL-encode the password automatically
- Test the connection (requires your IP whitelisted in Atlas)
- Save `MONGODB_URI="mongodb+srv://..."` to `.env` if successful

Manual password encoding examples:
`@` → `%40` | `?` → `%3F` | `:` → `%3A` | `/` → `%2F` | `#` → `%23` | `&` → `%26` | `=` → `%3D`

All application code now uses `mongodb_utils.get_db()` for access to the `facial_landmarks_db` database.

## 📚 Usage

### Step 1: Train the Model

Open and run the `new.ipynb` notebook:

```powershell
jupyter notebook new.ipynb
```

Or use VS Code to open the notebook directly.

The notebook includes:
1. **Data Loading**: Loads images and labels from train/val/test directories
2. **Model Architecture**: BiSeNet with ResNet18 backbone
3. **Training Loop**: 30 epochs with validation
4. **Visualization**: Training history plots

**Training Configuration:**
- Batch size: 4
- Learning rate: 0.01
- Optimizer: SGD with momentum
- Epochs: 30
- Classes: 11 (background + 10 landmarks)

After training completes, the best model will be saved as `best_model.pth`.

### Step 2: Run the Web Application

Once training is complete and `best_model.pth` exists, launch the web server:

```powershell
python landmark_app.py
```

You should see output like:
```
======================================================================
Facial Landmark Generation Web Application
======================================================================
Device: cuda (or cpu)
Model: BiSeNet with 11 landmark classes
Server starting on http://localhost:5000
======================================================================
```

### Step 3: Access the Web Interface

Open your web browser and navigate to:
```
http://localhost:5000
```

### Step 4: Generate Landmarks

1. **Upload Image**: 
   - Click the upload area or drag-and-drop an image
   - Supported formats: JPG, PNG (max 16MB)

2. **Generate**: 
   - Click "Generate Landmarks" button
   - Wait for processing (usually 1-3 seconds)

3. **View Results**:
   - Original image
   - Colored landmark mask
   - Overlay visualization
   - Landmark statistics (pixel counts per class)

## 🎨 Landmark Color Legend

| Landmark | Color |
|----------|-------|
| Background | Black |
| Landmark 1 | Red |
| Landmark 2 | Green |
| Landmark 3 | Blue |
| Landmark 4 | Yellow |
| Landmark 5 | Magenta |
| Landmark 6 | Cyan |
| Landmark 7 | Orange |
| Landmark 8 | Purple |
| Landmark 9 | Spring Green |
| Landmark 10 | Pink |

## 🔧 Model Architecture

**BiSeNet (Bilateral Segmentation Network)**

The model consists of:
- **Context Path**: ResNet18 backbone with attention refinement modules
- **Spatial Path**: Lightweight path for spatial information
- **Feature Fusion Module**: Combines context and spatial features
- **Output Heads**: Main output + 2 auxiliary outputs for training

**Total Parameters**: ~12.3M

## 📊 Training Tips

1. **Dataset Size**: 
   - Minimum 1000 images recommended
   - Current setup works with provided train/val/test split

2. **GPU Usage**:
   - Training is much faster with GPU
   - CPU training is possible but slower

3. **Batch Size**:
   - Adjust based on your GPU memory
   - Default: 4 (works on 8GB GPU)
   - Reduce to 2 if you get OOM errors

4. **Epochs**:
   - Default: 30 epochs
   - Increase for better accuracy
   - Monitor validation loss to avoid overfitting

## 🚀 API Endpoints

### GET `/`
- Returns the main HTML interface

### POST `/predict`
- Accepts image file upload
- Returns JSON with base64-encoded images and statistics

**Request:**
```
POST /predict
Content-Type: multipart/form-data
Body: image file
```

**Response:**
```json
{
  "success": true,
  "original": "data:image/png;base64,...",
  "prediction": "data:image/png;base64,...",
  "overlay": "data:image/png;base64,...",
  "landmark_counts": {
    "1": 15234,
    "2": 8456,
    ...
  },
  "image_size": [512, 512]
}
```

### GET `/health`
- Health check endpoint
- Returns model status

## 🐛 Troubleshooting

### Issue: Model not found
**Solution**: Make sure to run the training notebook first to generate `best_model.pth`

### Issue: Out of memory during training
**Solution**: Reduce batch size in the notebook (change `batch_size = 4` to `batch_size = 2`)

### Issue: Slow inference
**Solution**: 
- Use GPU if available
- Check that CUDA is properly installed
- Reduce image resolution (modify transform in app)

### Issue: Port 5000 already in use
**Solution**: Change port in `landmark_app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Use different port
```

## 📝 Notes

- The model requires `best_model.pth` to be present. If not found, it will use an untrained model (results will be poor).
- Training time depends on dataset size and hardware (~2-3 hours on GPU for 5000 images).
- The web app runs on localhost only by default. For production deployment, additional configuration is needed.

## 🤝 Contributing

Feel free to improve the project:
- Add more visualization options
- Implement additional model architectures (U-Net, DeepLab)
- Add batch processing capability
- Improve UI/UX

## 📄 License

This project is for educational purposes.

## 👨‍💻 Author

Capstone 4-1 Project

---

**Enjoy generating facial landmarks! 🎭**

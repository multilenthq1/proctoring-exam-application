# Proctoring Exam Application

A comprehensive AI-powered proctoring system for online examinations that monitors candidates in real-time using computer vision and audio analysis. This application leverages MediaPipe and OpenCV to detect suspicious behavior during exam sessions.

## 🎯 Features

### 1. **Head Pose Detection**
- Monitors the orientation of the candidate's head in real-time
- Tracks pitch (up/down), yaw (left/right), and roll angles
- Detects when the candidate is looking away from the screen
- Threshold-based detection for suspicious head movements
- Visual feedback with nose direction indicator

**Technology**: MediaPipe Face Mesh

### 2. **Eye Tracking**
- Tracks eye movements and gaze direction
- Detects iris position to determine where the candidate is looking
- Identifies suspicious gaze patterns (looking left, right, up, down)
- Monitors eye closure using Eye Aspect Ratio (EAR)
- Real-time gaze direction display

**Technology**: MediaPipe Face Mesh with iris landmarks

### 3. **Noise Detection**
- Monitors ambient noise levels in real-time
- Detects unauthorized voices or suspicious sounds
- Uses decibel (dB) measurements with configurable thresholds
- Smooths noise readings to reduce false positives
- Logs noise violations with dB levels

**Technology**: PyAudio with RMS-to-dB conversion

### 4. **Object Detection**
- Detects hands and their positions using hand landmark detection
- Identifies multiple faces in the frame (potential unauthorized person)
- Detects phone-like rectangular objects using contour analysis
- Monitors for unauthorized materials (books, phones, additional monitors)
- Provides real-time alerts for detected objects

**Technology**: MediaPipe Hands + OpenCV Haar Cascades + Contour Detection

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- Webcam/camera device
- Microphone
- Operating System: Windows, macOS, or Linux

### Step 1: Clone the Repository
```bash
git clone https://github.com/multilenthq1/proctoring-exam-application.git
cd proctoring-exam-application
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Note for PyAudio Installation:**
- **Windows**: PyAudio may require Microsoft C++ Build Tools. If installation fails, download a precompiled wheel from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio)
- **macOS**: You may need to install PortAudio first: `brew install portaudio`
- **Linux**: Install PortAudio development files: `sudo apt-get install portaudio19-dev`

### Step 4: Verify Installation
```bash
python -c "import cv2, mediapipe, pyaudio, numpy; print('All dependencies installed successfully!')"
```

## 📖 Usage

### Running the Application
```bash
python proctoring_app.py
```

### During the Exam Session
1. The application will open your default camera
2. Ensure your face is clearly visible and well-lit
3. The application will display real-time monitoring information:
   - Head pose angles (pitch, yaw, roll)
   - Gaze direction
   - Detected faces, hands, and objects
   - Ambient noise level (dB)
   - Violation count

4. All violations are logged to `violations.txt` with timestamps
5. Press `q` to quit the application

### Understanding the Display
- **Green text/indicators**: Normal behavior
- **Red text/indicators**: Suspicious behavior detected
- **Violation count**: Total number of violations detected during the session

## 📁 Project Structure

```
proctoring-exam-application/
├── src/
│   ├── head_pose_detector.py   # Head pose detection module
│   ├── eye_tracker.py           # Eye tracking module
│   ├── noise_detector.py        # Noise detection module
│   └── object_detector.py       # Object detection module
├── proctoring_app.py            # Main application
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── .gitignore                   # Git ignore file
└── violations.txt               # Auto-generated violation log
```

## 🔧 Configuration

### Adjusting Sensitivity

You can modify the detection thresholds in each module:

**Head Pose Detection** (`src/head_pose_detector.py`):
```python
self.pitch_threshold = 20  # Degrees (default: 20)
self.yaw_threshold = 25    # Degrees (default: 25)
```

**Eye Tracking** (`src/eye_tracker.py`):
```python
self.gaze_threshold = 0.15  # Ratio (default: 0.15)
self.eye_aspect_ratio_threshold = 0.2  # EAR (default: 0.2)
```

**Noise Detection** (`src/noise_detector.py`):
```python
# In ProctoringExamApp.__init__()
self.noise_detector = NoiseDetector(threshold_db=50)  # dB (default: 50)
```

## 🎥 Camera Settings

The application automatically configures the camera for optimal performance:
- Resolution: 1280x720
- Frame Rate: 30 FPS

To change the camera source, modify the `camera_index` parameter:
```python
app.start_camera(camera_index=0)  # 0 for default camera, 1 for secondary
```

## 📊 Output Files

### violations.txt
Contains a timestamped log of all violations detected during the exam session:
```
[2024-01-15 10:30:45] HEAD_POSE: Pitch: 25, Yaw: -30
[2024-01-15 10:31:10] EYE_TRACKING: Gaze Direction: LEFT
[2024-01-15 10:31:45] NOISE_LEVEL: Noise: 65 dB
[2024-01-15 10:32:20] OBJECT_DETECTION: Faces: 2, Objects: 1
```

At the end of the session, a summary is appended:
```
SESSION SUMMARY
================
Session Duration: 1200 seconds
Total Frames Processed: 36000
Total Violations: 15
Session Ended: 2024-01-15 11:00:00
```

## 🔍 How It Works

### Head Pose Detection
1. MediaPipe Face Mesh detects 468 facial landmarks
2. Key landmarks are used for 3D pose estimation via PnP algorithm
3. Rotation angles (pitch, yaw, roll) are calculated
4. Angles exceeding thresholds trigger violations

### Eye Tracking
1. Face mesh identifies eye contours and iris landmarks
2. Iris position relative to eye boundaries determines gaze
3. Eye Aspect Ratio (EAR) detects eye closure
4. Gaze direction is classified (CENTER, LEFT, RIGHT, UP, DOWN)

### Noise Detection
1. PyAudio captures audio in real-time chunks
2. Root Mean Square (RMS) is calculated from audio data
3. RMS is converted to decibels (dB)
4. Smoothing filter reduces false positives
5. Noise levels exceeding threshold trigger violations

### Object Detection
1. MediaPipe Hands detects hand landmarks and positions
2. OpenCV Haar Cascades detect faces (alerts if multiple faces)
3. Contour detection identifies rectangular objects (phones, tablets)
4. Combined analysis determines suspicious objects

## ⚠️ Limitations

- Requires adequate lighting for accurate face detection
- Performance depends on system hardware (CPU/GPU)
- PyAudio may not work on all audio devices
- Contour-based object detection may produce false positives
- Works best with a single person in frame

## 🛠️ Troubleshooting

### Camera Not Opening
- Check if another application is using the camera
- Try different camera index: `app.start_camera(camera_index=1)`
- Verify camera permissions in system settings

### PyAudio Errors
- Ensure microphone is connected and enabled
- Check microphone permissions in system settings
- The application will continue without audio monitoring if PyAudio fails

### Low FPS / Performance Issues
- Close other applications to free up resources
- Reduce camera resolution in `start_camera()` method
- Disable features you don't need

### MediaPipe Import Errors
- Ensure you're using Python 3.7-3.11 (MediaPipe compatibility)
- Try reinstalling: `pip uninstall mediapipe && pip install mediapipe==0.10.9`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- [MediaPipe](https://google.github.io/mediapipe/) by Google for face mesh and hand detection
- [OpenCV](https://opencv.org/) for computer vision utilities
- [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/) for audio capture

## 📧 Contact

For questions or support, please open an issue in the GitHub repository.

---

**Note**: This application is designed for educational and demonstration purposes. For production use in actual exam scenarios, additional security measures, backend integration, and compliance with privacy regulations should be implemented.
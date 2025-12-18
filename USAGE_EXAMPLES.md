# Usage Examples - Proctoring Exam Application

## Basic Usage

### 1. First Time Setup
```bash
# Clone the repository
git clone https://github.com/multilenthq1/proctoring-exam-application.git
cd proctoring-exam-application

# Install dependencies
pip install -r requirements.txt

# Verify installation
python verify_installation.py
```

### 2. Running the Application
```bash
# Start the proctoring application
python proctoring_app.py
```

## Customization Examples

### Example 1: Adjust Head Pose Sensitivity
Edit `src/head_pose_detector.py`:
```python
def __init__(self):
    # ... existing code ...
    
    # Make less sensitive (allow more head movement)
    self.pitch_threshold = 30  # Default: 20
    self.yaw_threshold = 35    # Default: 25
```

### Example 2: Change Noise Threshold
Edit `proctoring_app.py`:
```python
def __init__(self):
    # ... existing code ...
    
    # Make less sensitive to noise (higher threshold)
    self.noise_detector = NoiseDetector(threshold_db=60)  # Default: 50
```

### Example 3: Adjust Eye Tracking Sensitivity
Edit `src/eye_tracker.py`:
```python
def __init__(self):
    # ... existing code ...
    
    # Make less sensitive to gaze changes
    self.gaze_threshold = 0.20  # Default: 0.15 (higher = less sensitive)
```

### Example 4: Use Different Camera
Edit `proctoring_app.py`:
```python
def run(self):
    # ... existing code ...
    
    # Use secondary camera
    if not self.start_camera(camera_index=1):  # Default: 0
        print("Failed to start camera. Exiting...")
        return
```

### Example 5: Custom Object Detection Thresholds
Edit `src/object_detector.py`:
```python
def __init__(self):
    # ... existing code ...
    
    # Adjust object detection parameters
    self.min_object_area = 2000     # Default: 1000
    self.max_object_area = 40000    # Default: 50000
    self.min_aspect_ratio = 0.3     # Default: 0.4
    self.max_aspect_ratio = 0.8     # Default: 0.7
```

## Integration Examples

### Example 6: Programmatic Usage
```python
from src.head_pose_detector import HeadPoseDetector
from src.eye_tracker import EyeTracker
import cv2

# Initialize detectors
head_detector = HeadPoseDetector()
eye_tracker = EyeTracker()

# Open camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect head pose
    frame, pitch, yaw, roll, head_suspicious = head_detector.detect_head_pose(frame)
    
    # Track eyes
    frame, gaze, eye_suspicious = eye_tracker.track_eyes(frame)
    
    # Process results
    if head_suspicious or eye_suspicious:
        print(f"Suspicious behavior detected!")
    
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
head_detector.release()
eye_tracker.release()
```

### Example 7: Violation Callback
Modify `proctoring_app.py` to add custom violation handling:
```python
class ProctoringExamApp:
    def __init__(self, violation_callback=None):
        # ... existing code ...
        self.violation_callback = violation_callback
    
    def log_violation(self, violation_type, details):
        # ... existing code ...
        
        # Call custom callback if provided
        if self.violation_callback:
            self.violation_callback(violation_type, details)

# Usage
def my_violation_handler(v_type, details):
    print(f"ALERT: {v_type} - {details}")
    # Send email, SMS, or save to database

app = ProctoringExamApp(violation_callback=my_violation_handler)
app.run()
```

## Testing Examples

### Example 8: Test Individual Modules
```python
# Test head pose detection only
from src.head_pose_detector import HeadPoseDetector
import cv2

detector = HeadPoseDetector()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame, pitch, yaw, roll, suspicious = detector.detect_head_pose(frame)
    print(f"Pitch: {pitch:.2f}, Yaw: {yaw:.2f}, Roll: {roll:.2f}")
    
    cv2.imshow('Head Pose', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
detector.release()
```

### Example 9: Test Noise Detection Without Video
```python
from src.noise_detector import NoiseDetector
import time

detector = NoiseDetector(threshold_db=50)
detector.start_monitoring()

print("Monitoring noise for 30 seconds...")
for i in range(30):
    noise_level, suspicious = detector.detect_noise()
    status = "SUSPICIOUS" if suspicious else "NORMAL"
    print(f"Second {i+1}: {noise_level:.2f} dB - {status}")
    time.sleep(1)

detector.stop_monitoring()
```

## Performance Optimization Examples

### Example 10: Reduce Frame Processing Rate
```python
# Process every Nth frame for better performance
class ProctoringExamApp:
    def __init__(self):
        # ... existing code ...
        self.process_every_n_frames = 2  # Process every 2nd frame
    
    def process_frame(self, frame):
        if self.frame_count % self.process_every_n_frames != 0:
            return frame  # Skip processing
        
        # ... rest of processing ...
```

### Example 11: Disable Specific Features
```python
# Disable object detection for better performance
class ProctoringExamApp:
    def __init__(self, enable_object_detection=False):
        # ... existing code ...
        
        if enable_object_detection:
            self.object_detector = ObjectDetector()
        else:
            self.object_detector = None
    
    def process_frame(self, frame):
        # ... existing code ...
        
        # Only run object detection if enabled
        if self.object_detector:
            frame, detection_info, object_suspicious = \
                self.object_detector.detect_objects(frame)
```

## Output Examples

### Example Violation Log (violations.txt)
```
Exam Session Started: 2024-12-18 10:00:00
============================================================

[2024-12-18 10:01:23] HEAD_POSE: Pitch: 25, Yaw: -30
[2024-12-18 10:02:45] EYE_TRACKING: Gaze Direction: LEFT
[2024-12-18 10:03:12] NOISE_LEVEL: Noise: 65 dB
[2024-12-18 10:04:56] OBJECT_DETECTION: Faces: 2, Objects: 0
[2024-12-18 10:05:34] HEAD_POSE: Pitch: -22, Yaw: 15
[2024-12-18 10:06:18] EYE_TRACKING: Gaze Direction: RIGHT

============================================================
SESSION SUMMARY
============================================================
Session Duration: 3600 seconds
Total Frames Processed: 108000
Total Violations: 6
Session Ended: 2024-12-18 11:00:00
```

## Tips and Best Practices

1. **For Testing**: Start with lenient thresholds and adjust based on environment
2. **For Production**: Use stricter thresholds and test thoroughly
3. **Performance**: Disable features you don't need
4. **Lighting**: Ensure good lighting for accurate face detection
5. **Audio**: Test audio monitoring in actual exam environment
6. **Privacy**: Ensure compliance with local privacy regulations

## Troubleshooting Common Issues

### Issue: Camera Not Found
```python
# Try all available cameras
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera found at index {i}")
        cap.release()
```

### Issue: PyAudio Not Working
```python
# The application will continue without audio if PyAudio fails
# Check in the console for "Noise monitoring started..."
# If not present, audio monitoring is disabled
```

### Issue: Low FPS
```python
# Reduce camera resolution
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Default: 1280
self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Default: 720
```

---

For more information, see:
- `README.md` - Full documentation
- `QUICKSTART.md` - Quick start guide
- `IMPLEMENTATION_SUMMARY.md` - Technical details

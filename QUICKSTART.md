# Quick Start Guide

## Getting Started with Proctoring Exam Application

### 1. Installation

#### Step 1: Install Python Dependencies
```bash
pip install -r requirements.txt
```

#### Step 2: Verify Installation
```bash
python verify_installation.py
```

If you see "VERIFICATION PASSED", you're ready to go!

### 2. Running the Application

#### Basic Usage
```bash
python proctoring_app.py
```

#### What to Expect
- A window will open showing your camera feed
- Real-time monitoring information will be displayed:
  - Head pose angles (Pitch, Yaw, Roll)
  - Gaze direction (CENTER, LEFT, RIGHT, UP, DOWN)
  - Number of detected faces, hands, and objects
  - Ambient noise level in decibels (dB)
  - Total violation count

#### During the Session
- Keep your face visible and centered
- Look at the screen (camera)
- Avoid excessive head movements
- Keep the environment quiet
- Don't bring unauthorized materials into view

#### Ending the Session
- Press 'q' to quit
- Review `violations.txt` for logged violations
- Check the session summary printed in the console

### 3. Troubleshooting

#### Camera Issues
```bash
# Try a different camera index
# Edit proctoring_app.py line where start_camera() is called
app.start_camera(camera_index=1)  # Try 1, 2, etc.
```

#### PyAudio Installation Issues

**Windows:**
```bash
# If pip install fails, download wheel from:
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
# Then install:
pip install PyAudio-0.2.11-cp39-cp39-win_amd64.whl
```

**macOS:**
```bash
brew install portaudio
pip install pyaudio
```

**Linux:**
```bash
sudo apt-get install portaudio19-dev
pip install pyaudio
```

#### Performance Issues
- Close other applications
- Ensure good lighting
- Use a faster computer if available

### 4. Configuration

Edit the threshold values in the source files to adjust sensitivity:

**File: `src/head_pose_detector.py`**
```python
self.pitch_threshold = 20  # Increase to be less sensitive
self.yaw_threshold = 25    # Increase to be less sensitive
```

**File: `src/eye_tracker.py`**
```python
self.gaze_threshold = 0.15  # Increase to be less sensitive
```

**File: `proctoring_app.py`**
```python
self.noise_detector = NoiseDetector(threshold_db=50)  # Increase for louder threshold
```

### 5. Understanding the Output

#### On-Screen Display
- **Green text**: Normal behavior
- **Red text**: Suspicious behavior detected
- **Violation count**: Increases when suspicious activity is detected

#### violations.txt File
Contains timestamped violations:
```
[2024-01-15 10:30:45] HEAD_POSE: Pitch: 25, Yaw: -30
[2024-01-15 10:31:10] EYE_TRACKING: Gaze Direction: LEFT
[2024-01-15 10:31:45] NOISE_LEVEL: Noise: 65 dB
```

### 6. Best Practices

#### For Candidates
1. Sit in a well-lit room
2. Position camera at eye level
3. Keep face centered and visible
4. Look at the screen/camera
5. Minimize background noise
6. Don't use phone or other devices

#### For Administrators
1. Test the setup before the exam
2. Calibrate thresholds for your environment
3. Review violation logs after the exam
4. Consider false positives in final assessment

### 7. Known Limitations

- Requires good lighting
- May produce false positives
- PyAudio might not work on all systems
- Performance depends on hardware
- Works best with single person in frame

### 8. Getting Help

If you encounter issues:
1. Check the Troubleshooting section above
2. Review the full README.md
3. Run `python verify_installation.py`
4. Open an issue on GitHub

---

**Ready to start?** Run: `python proctoring_app.py`

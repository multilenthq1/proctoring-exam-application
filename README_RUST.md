# Proctoring Exam Application (Rust Version)

A Rust implementation of an intelligent exam proctoring system that monitors candidates during online exams using computer vision and audio processing.

## Features

- **Head Pose Detection**: Tracks head orientation to ensure candidates are looking at the screen
- **Eye Tracking**: Monitors eye movements and gaze direction
- **Object Detection**: Detects unauthorized objects like phones, books, and laptops
- **Noise Detection**: Monitors ambient noise levels to detect unauthorized voices
- **Real-time Violation Logging**: Records all suspicious activities with timestamps

## Requirements

### System Dependencies

#### macOS
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install OpenCV
brew install opencv

# Install pkg-config
brew install pkg-config

# Install clang (usually pre-installed on macOS)
xcode-select --install
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install -y \
    libopencv-dev \
    clang \
    libclang-dev \
    pkg-config \
    libasound2-dev
```

#### Windows
1. Install Visual Studio 2019 or later with C++ tools
2. Download and install OpenCV from https://opencv.org/releases/
3. Set environment variable `OPENCV_DIR` to OpenCV installation path
4. Install LLVM/Clang from https://releases.llvm.org/

### Rust

Install Rust using rustup:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### OpenCV Haar Cascade Files

Download required cascade files (if not already present):
```bash
# Face detection
curl -o haarcascade_frontalface_default.xml \
  https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml

# Eye detection
curl -o haarcascade_eye.xml \
  https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml
```

## Building

1. Clone the repository:
```bash
cd /Users/buki/proctoring-exam-application
```

2. Build the project:
```bash
cargo build --release
```

## Running

Run the application:
```bash
cargo run --release
```

Or run the binary directly:
```bash
./target/release/proctoring-exam-app
```

## Usage

1. Launch the application
2. Allow camera and microphone access when prompted
3. Position yourself in front of the camera
4. The application will monitor:
   - Head orientation
   - Eye movements
   - Faces in view
   - Objects in frame
   - Ambient noise levels

5. Press 'q' to quit the application

All violations are logged to `violations.txt` with timestamps.

## Project Structure

```
proctoring-exam-application/
├── Cargo.toml                      # Rust project configuration
├── src/
│   ├── main.rs                     # Main application entry point
│   ├── head_pose_detector.rs       # Head pose detection module
│   ├── eye_tracker.rs              # Eye tracking module
│   ├── noise_detector.rs           # Noise detection module
│   └── object_detector.rs          # Object detection module
├── haarcascade_frontalface_default.xml
├── haarcascade_eye.xml
└── README_RUST.md                  # This file
```

## Differences from Python Version

### Libraries Used

| Feature | Python | Rust |
|---------|--------|------|
| Computer Vision | OpenCV + MediaPipe | OpenCV |
| Audio Processing | PyAudio | cpal + dasp |
| Deep Learning | MediaPipe Tasks | OpenCV DNN (YOLO) |
| Face Detection | MediaPipe Face Mesh | Haar Cascades |
| Eye Tracking | MediaPipe Iris | Haar Cascades + Custom Logic |

### Key Changes

1. **Face/Eye Detection**: The Rust version uses OpenCV's Haar Cascade classifiers instead of MediaPipe, as MediaPipe doesn't have official Rust bindings. This provides similar functionality but with different accuracy characteristics.

2. **Audio Processing**: Uses `cpal` for cross-platform audio input instead of PyAudio.

3. **Performance**: The Rust version typically has:
   - Lower memory usage
   - Faster execution
   - Better CPU efficiency
   - Smaller binary size (when stripped)

4. **Error Handling**: Uses Rust's `Result` type for robust error handling instead of Python exceptions.

5. **Concurrency**: Audio processing runs on a separate thread using Rust's safe concurrency primitives.

## Performance Optimization

For best performance:

1. **Release Build**: Always use `--release` flag
```bash
cargo build --release
```

2. **Optimize Binary Size** (optional):
Add to `Cargo.toml`:
```toml
[profile.release]
opt-level = 'z'     # Optimize for size
lto = true          # Link-time optimization
codegen-units = 1   # Better optimization
strip = true        # Remove debug symbols
```

3. **GPU Acceleration**: If available, OpenCV will use GPU acceleration for DNN operations.

## Troubleshooting

### OpenCV Not Found
```bash
# macOS
export PKG_CONFIG_PATH="/opt/homebrew/opt/opencv/lib/pkgconfig"

# Linux
export PKG_CONFIG_PATH="/usr/lib/x86_64-linux-gnu/pkgconfig"
```

### Audio Issues
- **macOS**: Grant microphone permissions in System Preferences > Security & Privacy
- **Linux**: Ensure you're in the `audio` group: `sudo usermod -a -G audio $USER`
- **Windows**: Check Windows Privacy Settings for microphone access

### Cascade Files Not Found
Ensure the `.xml` files are in the same directory as the executable or provide absolute paths in the code.

## Limitations

1. **MediaPipe Features**: Some advanced MediaPipe features (like iris tracking) are not available in the Rust version. Consider creating FFI bindings to MediaPipe C++ API for full feature parity.

2. **Hand Detection**: The current version has limited hand detection. For production use, consider integrating YOLO or other hand detection models.

3. **Model Files**: YOLO model files (`yolov3-tiny.cfg` and `yolov3-tiny.weights`) are needed for object detection.

## Future Enhancements

- [ ] MediaPipe FFI bindings for better face/eye tracking
- [ ] GPU acceleration for YOLO inference
- [ ] Web interface using WebAssembly
- [ ] Docker containerization
- [ ] Cloud deployment options

## License

Same as the original Python version.

## Conversion Notes

This Rust version maintains functional parity with the Python implementation while leveraging Rust's:
- Memory safety without garbage collection
- Zero-cost abstractions
- Fearless concurrency
- Type safety
- Performance characteristics

For questions or contributions, please refer to the main project documentation.

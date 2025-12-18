"""
Installation Verification Script
Run this script after installing dependencies to verify the setup.
"""

import sys

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("Checking dependencies...\n")
    
    dependencies = {
        'cv2': 'opencv-python',
        'mediapipe': 'mediapipe',
        'numpy': 'numpy',
        'pyaudio': 'pyaudio'
    }
    
    missing = []
    installed = []
    
    for module, package in dependencies.items():
        try:
            __import__(module)
            installed.append(f"✓ {package}")
        except ImportError:
            missing.append(f"✗ {package}")
    
    # Display results
    if installed:
        print("Installed packages:")
        for pkg in installed:
            print(f"  {pkg}")
    
    if missing:
        print("\nMissing packages:")
        for pkg in missing:
            print(f"  {pkg}")
        print("\nPlease install missing packages using:")
        print("  pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All dependencies are installed!")
        return True


def check_modules():
    """Check if all proctoring modules can be imported."""
    print("\nChecking application modules...\n")
    
    modules = [
        'src.head_pose_detector',
        'src.eye_tracker',
        'src.noise_detector',
        'src.object_detector'
    ]
    
    for module in modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError as e:
            print(f"  ✗ {module}: {e}")
            return False
    
    print("\n✓ All application modules are working!")
    return True


def check_camera():
    """Check if camera is available."""
    print("\nChecking camera access...\n")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                print("  ✓ Camera is accessible and working")
                return True
            else:
                print("  ✗ Camera opened but failed to read frame")
                return False
        else:
            print("  ✗ Failed to open camera")
            print("  Make sure your camera is connected and not in use")
            return False
            
    except Exception as e:
        print(f"  ✗ Camera check failed: {e}")
        return False


def check_audio():
    """Check if audio input is available."""
    print("\nChecking audio input...\n")
    
    try:
        import pyaudio
        audio = pyaudio.PyAudio()
        
        # Check if there are input devices
        device_count = audio.get_device_count()
        input_devices = []
        
        for i in range(device_count):
            device_info = audio.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                input_devices.append(device_info['name'])
        
        audio.terminate()
        
        if input_devices:
            print(f"  ✓ Found {len(input_devices)} audio input device(s)")
            return True
        else:
            print("  ✗ No audio input devices found")
            print("  Audio monitoring may not work")
            return False
            
    except Exception as e:
        print(f"  ✗ Audio check failed: {e}")
        print("  Note: Application will work without audio monitoring")
        return False


def main():
    """Main verification function."""
    print("="*60)
    print("PROCTORING EXAM APPLICATION - INSTALLATION VERIFICATION")
    print("="*60 + "\n")
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\n" + "="*60)
        print("VERIFICATION FAILED: Missing dependencies")
        print("="*60)
        sys.exit(1)
    
    # Check modules
    modules_ok = check_modules()
    
    if not modules_ok:
        print("\n" + "="*60)
        print("VERIFICATION FAILED: Module import errors")
        print("="*60)
        sys.exit(1)
    
    # Check camera (warning only)
    camera_ok = check_camera()
    
    # Check audio (warning only)
    audio_ok = check_audio()
    
    # Summary
    print("\n" + "="*60)
    if modules_ok and camera_ok and audio_ok:
        print("VERIFICATION PASSED: All systems are ready!")
        print("\nYou can now run the application:")
        print("  python proctoring_app.py")
    elif modules_ok:
        print("VERIFICATION PASSED: Core modules are ready!")
        if not camera_ok:
            print("\nWarning: Camera check failed")
        if not audio_ok:
            print("Warning: Audio check failed")
        print("\nYou can still try running the application:")
        print("  python proctoring_app.py")
    print("="*60)


if __name__ == "__main__":
    main()

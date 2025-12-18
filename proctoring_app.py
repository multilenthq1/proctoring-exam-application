"""
Proctoring Exam Application - Main Module
Integrates all proctoring features: head pose detection, noise detection,
eye tracking, and object detection.
"""

import cv2
import sys
import time
from datetime import datetime

# Import proctoring modules
from src.head_pose_detector import HeadPoseDetector
from src.noise_detector import NoiseDetector
from src.eye_tracker import EyeTracker
from src.object_detector import ObjectDetector


class ProctoringExamApp:
    """
    Main application class that integrates all proctoring features.
    """
    
    def __init__(self):
        """Initialize all proctoring modules."""
        print("Initializing Proctoring Exam Application...")
        
        # Initialize detectors
        self.head_pose_detector = HeadPoseDetector()
        self.noise_detector = NoiseDetector(threshold_db=50)
        self.eye_tracker = EyeTracker()
        self.object_detector = ObjectDetector()
        
        # Initialize video capture
        self.cap = None
        
        # Violation tracking
        self.violations = []
        self.violation_file = "violations.txt"
        
        # Session info
        self.session_start_time = None
        self.frame_count = 0
        
        print("Initialization complete!")
    
    def start_camera(self, camera_index=0):
        """
        Start the camera feed.
        
        Args:
            camera_index: Index of the camera to use (default: 0)
        """
        self.cap = cv2.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return False
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"Camera {camera_index} started successfully")
        return True
    
    def log_violation(self, violation_type, details):
        """
        Log a violation to the violations list and file.
        
        Args:
            violation_type: Type of violation
            details: Additional details about the violation
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        violation_entry = f"[{timestamp}] {violation_type}: {details}"
        
        self.violations.append(violation_entry)
        
        # Write to file
        with open(self.violation_file, 'a') as f:
            f.write(violation_entry + '\n')
        
        print(f"VIOLATION LOGGED: {violation_entry}")
    
    def process_frame(self, frame):
        """
        Process a single frame through all detection modules.
        
        Args:
            frame: Input video frame
            
        Returns:
            Annotated frame with all detection results
        """
        self.frame_count += 1
        
        # Head Pose Detection
        frame, pitch, yaw, roll, head_suspicious = self.head_pose_detector.detect_head_pose(frame)
        if head_suspicious:
            self.log_violation(
                "HEAD_POSE",
                f"Pitch: {int(pitch)}, Yaw: {int(yaw)}"
            )
        
        # Eye Tracking
        frame, gaze_direction, eye_suspicious = self.eye_tracker.track_eyes(frame)
        if eye_suspicious:
            self.log_violation(
                "EYE_TRACKING",
                f"Gaze Direction: {gaze_direction}"
            )
        
        # Object Detection
        frame, detection_info, object_suspicious = self.object_detector.detect_objects(frame)
        if object_suspicious:
            self.log_violation(
                "OBJECT_DETECTION",
                f"Faces: {detection_info['num_faces']}, Objects: {detection_info['num_objects']}"
            )
        
        # Noise Detection (non-blocking)
        noise_level, noise_suspicious = self.noise_detector.detect_noise()
        if noise_suspicious:
            self.log_violation(
                "NOISE_LEVEL",
                f"Noise: {int(noise_level)} dB"
            )
        
        # Display noise level on frame
        noise_color = (0, 0, 255) if noise_suspicious else (0, 255, 0)
        cv2.putText(frame, f"Noise: {int(noise_level)} dB", (20, 350), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, noise_color, 2)
        
        # Display session information
        elapsed_time = time.time() - self.session_start_time
        cv2.putText(frame, f"Session Time: {int(elapsed_time)}s", (20, 390), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Violations: {len(self.violations)}", (20, 420), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Display instructions
        cv2.putText(frame, "Press 'q' to quit", (frame.shape[1] - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """
        Main application loop.
        Run the proctoring application.
        """
        print("\n" + "="*60)
        print("PROCTORING EXAM APPLICATION")
        print("="*60)
        print("\nStarting exam session...")
        print("Please ensure:")
        print("  - You are in a well-lit room")
        print("  - Your face is clearly visible")
        print("  - No unauthorized materials are present")
        print("\nPress 'q' to quit the application")
        print("="*60 + "\n")
        
        # Start camera
        if not self.start_camera():
            print("Failed to start camera. Exiting...")
            return
        
        # Start noise monitoring
        self.noise_detector.start_monitoring()
        
        # Initialize session
        self.session_start_time = time.time()
        
        # Clear previous violations file
        with open(self.violation_file, 'w') as f:
            f.write(f"Exam Session Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
        
        # Main processing loop
        try:
            while True:
                # Read frame from camera
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Error: Failed to capture frame")
                    break
                
                # Process frame through all detectors
                processed_frame = self.process_frame(frame)
                
                # Display the frame
                cv2.imshow('Proctoring Exam Application', processed_frame)
                
                # Check for quit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nExiting application...")
                    break
                
        except KeyboardInterrupt:
            print("\nApplication interrupted by user")
        
        finally:
            # Cleanup
            self.cleanup()
    
    def cleanup(self):
        """Release all resources and save session summary."""
        print("\nCleaning up resources...")
        
        # Release camera
        if self.cap:
            self.cap.release()
        
        # Stop noise monitoring
        self.noise_detector.stop_monitoring()
        
        # Release detectors
        self.head_pose_detector.release()
        self.eye_tracker.release()
        self.object_detector.release()
        
        # Close windows
        cv2.destroyAllWindows()
        
        # Generate session summary
        session_duration = time.time() - self.session_start_time if self.session_start_time else 0
        
        print("\n" + "="*60)
        print("EXAM SESSION SUMMARY")
        print("="*60)
        print(f"Session Duration: {int(session_duration)} seconds")
        print(f"Total Frames Processed: {self.frame_count}")
        print(f"Total Violations: {len(self.violations)}")
        print(f"Violations logged to: {self.violation_file}")
        print("="*60 + "\n")
        
        # Write summary to violations file
        with open(self.violation_file, 'a') as f:
            f.write("\n" + "="*60 + "\n")
            f.write("SESSION SUMMARY\n")
            f.write("="*60 + "\n")
            f.write(f"Session Duration: {int(session_duration)} seconds\n")
            f.write(f"Total Frames Processed: {self.frame_count}\n")
            f.write(f"Total Violations: {len(self.violations)}\n")
            f.write(f"Session Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


def main():
    """Main entry point for the application."""
    app = ProctoringExamApp()
    app.run()


if __name__ == "__main__":
    main()

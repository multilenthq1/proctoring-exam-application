"""
Head Pose Detection Module
Uses MediaPipe Face Mesh to detect head orientation and track candidate's focus.
"""

import cv2
import mediapipe as mp
import numpy as np


class HeadPoseDetector:
    """
    Detects head pose using MediaPipe Face Mesh.
    Tracks head orientation to ensure candidate is looking at the screen.
    """
    
    def __init__(self):
        """Initialize MediaPipe Face Mesh for head pose detection."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        
        # Thresholds for direction detection (ratio-based)
        self.horizontal_threshold = 0.03  # For left/right
        self.vertical_threshold = 0.02    # For up/down
        
        # Current detected direction
        self.current_direction = "Forward"
        
    def detect_head_pose(self, frame):
        """
        Detect head pose from a video frame using nose position relative to face.
        
        Args:
            frame: Input BGR image frame
            
        Returns:
            tuple: (annotated_frame, pitch, yaw, roll, is_suspicious, direction)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        is_suspicious = False
        pitch, yaw, roll = 0, 0, 0
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get image dimensions
                img_h, img_w, _ = frame.shape
                
                # Key landmark indices in MediaPipe Face Mesh:
                # 1 = Nose tip
                # 33 = Right eye outer corner
                # 263 = Left eye outer corner  
                # 61 = Right mouth corner
                # 291 = Left mouth corner
                # 199 = Chin bottom
                # 10 = Forehead top
                # 152 = Chin
                
                # Get nose tip position
                nose_tip = face_landmarks.landmark[1]
                nose_x = nose_tip.x
                nose_y = nose_tip.y
                
                # Get face boundaries using eye corners and chin
                left_eye = face_landmarks.landmark[263]  # Left eye outer
                right_eye = face_landmarks.landmark[33]   # Right eye outer
                chin = face_landmarks.landmark[152]
                forehead = face_landmarks.landmark[10]
                
                # Calculate face center (between eyes)
                face_center_x = (left_eye.x + right_eye.x) / 2
                face_center_y = (forehead.y + chin.y) / 2
                
                # Calculate face width and height for normalization
                face_width = abs(left_eye.x - right_eye.x)
                face_height = abs(chin.y - forehead.y)
                
                # Calculate normalized offset of nose from center
                horizontal_offset = (nose_x - face_center_x) / face_width if face_width > 0 else 0
                vertical_offset = (nose_y - face_center_y) / face_height if face_height > 0 else 0
                
                # Determine direction
                directions = []
                
                # Horizontal: positive = looking left, negative = looking right
                if horizontal_offset < -self.horizontal_threshold:
                    directions.append("Right")
                    is_suspicious = True
                elif horizontal_offset > self.horizontal_threshold:
                    directions.append("Left")
                    is_suspicious = True
                
                # Vertical: positive = looking down, negative = looking up
                if vertical_offset < -self.vertical_threshold:
                    directions.append("Up")
                    is_suspicious = True
                elif vertical_offset > self.vertical_threshold:
                    directions.append("Down")
                    is_suspicious = True
                
                if directions:
                    self.current_direction = " + ".join(directions)
                else:
                    self.current_direction = "Forward"
                
                # Set display color based on status
                if is_suspicious:
                    color = (0, 0, 255)  # Red
                    status = "LOOKING AWAY"
                else:
                    color = (0, 255, 0)  # Green
                    status = "LOOKING AT SCREEN"
                
                # Draw face mesh (optional - for debugging)
                # self.mp_drawing.draw_landmarks(frame, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS)
                
                # Draw nose tip indicator
                nose_px = int(nose_x * img_w)
                nose_py = int(nose_y * img_h)
                cv2.circle(frame, (nose_px, nose_py), 5, color, -1)
                
                # Draw face center for reference
                center_px = int(face_center_x * img_w)
                center_py = int(face_center_y * img_h)
                cv2.circle(frame, (center_px, center_py), 5, (255, 255, 0), -1)
                
                # Draw line from center to nose
                cv2.line(frame, (center_px, center_py), (nose_px, nose_py), color, 2)
                
                # Display direction prominently
                cv2.putText(frame, f"Direction: {self.current_direction}", (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.putText(frame, f"Status: {status}", (20, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
        return frame, pitch, yaw, roll, is_suspicious, self.current_direction
    
    def release(self):
        """Release resources."""
        self.face_mesh.close()

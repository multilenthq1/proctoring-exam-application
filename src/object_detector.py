"""
Object Detection Module
Detects unauthorized objects in the camera feed using MediaPipe.
"""

import cv2
import mediapipe as mp
import numpy as np


class ObjectDetector:
    """
    Detects objects in camera feed to identify potential cheating aids.
    Uses MediaPipe object detection to identify phones, books, and other items.
    """
    
    def __init__(self):
        """Initialize object detection."""
        # For object detection, we'll use MediaPipe's solutions
        # Note: MediaPipe doesn't have a direct object detection for all objects,
        # so we'll use hand detection as a proxy for detecting when candidate
        # is holding or manipulating objects
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # For additional object detection, we can use simple color-based detection
        # or contour detection for phones, books, etc.
        self.suspicious_object_detected = False
        
    def detect_hands(self, frame):
        """
        Detect hands in the frame.
        
        Args:
            frame: Input BGR image frame
            
        Returns:
            tuple: (annotated_frame, num_hands, hand_positions)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        num_hands = 0
        hand_positions = []
        
        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)
            
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                # Get hand position (using wrist landmark)
                img_h, img_w, _ = frame.shape
                wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                x, y = int(wrist.x * img_w), int(wrist.y * img_h)
                hand_positions.append((x, y))
        
        return frame, num_hands, hand_positions
    
    def detect_multiple_faces(self, frame):
        """
        Detect if multiple faces are present (indicating another person).
        
        Args:
            frame: Input BGR image frame
            
        Returns:
            tuple: (num_faces, is_suspicious)
        """
        # Use OpenCV's Haar Cascade for face detection
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        num_faces = len(faces)
        is_suspicious = num_faces > 1 or num_faces == 0
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            color = (0, 0, 255) if is_suspicious else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        return num_faces, is_suspicious
    
    def detect_phone_like_objects(self, frame):
        """
        Detect phone-like rectangular objects using contour detection.
        
        Args:
            frame: Input BGR image frame
            
        Returns:
            tuple: (num_objects, is_suspicious)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        phone_like_objects = 0
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size and aspect ratio (typical phone dimensions)
            area = w * h
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Phone-like objects: rectangular, specific size range
            if 1000 < area < 50000 and 0.4 < aspect_ratio < 0.7:
                phone_like_objects += 1
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, "Possible Object", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        is_suspicious = phone_like_objects > 0
        return phone_like_objects, is_suspicious
    
    def detect_objects(self, frame):
        """
        Comprehensive object detection combining multiple techniques.
        
        Args:
            frame: Input BGR image frame
            
        Returns:
            tuple: (annotated_frame, detection_info, is_suspicious)
        """
        # Detect hands
        frame, num_hands, hand_positions = self.detect_hands(frame)
        
        # Detect multiple faces
        num_faces, faces_suspicious = self.detect_multiple_faces(frame)
        
        # Detect phone-like objects
        num_objects, objects_suspicious = self.detect_phone_like_objects(frame)
        
        # Overall suspicious status
        is_suspicious = faces_suspicious or objects_suspicious or num_hands > 2
        
        # Prepare detection info
        detection_info = {
            'num_hands': num_hands,
            'num_faces': num_faces,
            'num_objects': num_objects,
            'hand_positions': hand_positions
        }
        
        # Display object detection information
        color = (0, 0, 255) if is_suspicious else (0, 255, 0)
        status = "SUSPICIOUS" if is_suspicious else "NORMAL"
        
        cv2.putText(frame, f"Faces: {num_faces}", (20, 230), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"Hands: {num_hands}", (20, 260), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"Objects: {num_objects}", (20, 290), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"Object Status: {status}", (20, 320), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return frame, detection_info, is_suspicious
    
    def release(self):
        """Release resources."""
        self.hands.close()

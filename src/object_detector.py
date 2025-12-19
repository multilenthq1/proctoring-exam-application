"""
Object Detection Module
Detects unauthorized objects and multiple faces using MediaPipe Tasks API.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import urllib.request


class ObjectDetector:
    """
    Detects objects in camera feed to identify potential cheating aids.
    Uses MediaPipe Tasks API for object detection (phones, etc.) and face/hand detection.
    """
    
    def __init__(self):
        """Initialize MediaPipe detection modules."""
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )
        
        # Drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize MediaPipe Tasks Object Detector for phone detection
        self.object_detector = None
        self._init_object_detector()
        
        # Detection state
        self.phone_detected = False
        self.detected_objects = []
        
        # Suspicious object categories (from COCO dataset)
        self.suspicious_categories = ['cell phone', 'book', 'laptop', 'remote']
        
        print("MediaPipe Object Detector initialized!")
    
    def _init_object_detector(self):
        """Initialize MediaPipe Tasks Object Detector with EfficientDet model."""
        model_path = "efficientdet_lite0.tflite"
        
        # Download model if not present
        if not os.path.exists(model_path):
            print("Downloading EfficientDet model for phone detection...")
            try:
                urllib.request.urlretrieve(
                    "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite",
                    model_path
                )
                print("EfficientDet model downloaded successfully!")
            except Exception as e:
                print(f"Could not download model: {e}")
                return
        
        # Create object detector
        try:
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.ObjectDetectorOptions(
                base_options=base_options,
                score_threshold=0.5,
                max_results=5
            )
            self.object_detector = vision.ObjectDetector.create_from_options(options)
            print("MediaPipe Object Detector (EfficientDet) loaded!")
        except Exception as e:
            print(f"Could not initialize object detector: {e}")
            self.object_detector = None
    
    def detect_objects_efficientdet(self, frame):
        """
        Detect objects using MediaPipe Tasks EfficientDet model.
        
        Args:
            frame: Input BGR image frame
            
        Returns:
            tuple: (annotated_frame, detected_objects, phone_detected)
        """
        detected_objects = []
        phone_detected = False
        
        if self.object_detector is None:
            return frame, detected_objects, phone_detected
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect objects
        detection_result = self.object_detector.detect(mp_image)
        
        # Process detections
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            category = detection.categories[0]
            category_name = category.category_name
            score = category.score
            
            # Check if it's a suspicious object
            if category_name.lower() in [s.lower() for s in self.suspicious_categories]:
                detected_objects.append(category_name)
                
                if category_name.lower() == 'cell phone':
                    phone_detected = True
                
                # Draw bounding box (red for suspicious)
                start_point = (bbox.origin_x, bbox.origin_y)
                end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
                cv2.rectangle(frame, start_point, end_point, (0, 0, 255), 3)
                
                # Draw label
                label = f"{category_name.upper()} ({score:.2f})"
                cv2.putText(frame, label, (bbox.origin_x, bbox.origin_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Draw alert
                if category_name.lower() == 'cell phone':
                    cv2.putText(frame, "PHONE DETECTED!", (bbox.origin_x, bbox.origin_y - 35),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return frame, detected_objects, phone_detected
    
    def detect_hands(self, frame):
        """
        Detect hands in the frame using MediaPipe.
        
        Args:
            frame: Input BGR image frame
            
        Returns:
            tuple: (annotated_frame, num_hands, hand_positions)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        num_hands = 0
        hand_positions = []
        
        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)
            
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                )
                
                # Get hand position (using wrist landmark)
                img_h, img_w, _ = frame.shape
                wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                x, y = int(wrist.x * img_w), int(wrist.y * img_h)
                hand_positions.append((x, y))
                
                # Label the hand
                hand_label = handedness.classification[0].label
                cv2.putText(frame, hand_label, (x - 30, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame, num_hands, hand_positions
    
    def detect_faces(self, frame):
        """
        Detect faces using MediaPipe Face Detection.
        
        Args:
            frame: Input BGR image frame
            
        Returns:
            tuple: (annotated_frame, num_faces, is_suspicious)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        num_faces = 0
        is_suspicious = False
        
        img_h, img_w, _ = frame.shape
        
        if results.detections:
            num_faces = len(results.detections)
            
            # Suspicious if no face or multiple faces
            is_suspicious = num_faces != 1
            
            for detection in results.detections:
                # Get bounding box
                bboxC = detection.location_data.relative_bounding_box
                x = int(bboxC.xmin * img_w)
                y = int(bboxC.ymin * img_h)
                w = int(bboxC.width * img_w)
                h = int(bboxC.height * img_h)
                
                # Draw bounding box
                color = (0, 0, 255) if is_suspicious else (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Draw confidence score
                confidence = detection.score[0]
                cv2.putText(frame, f"Face: {confidence:.2f}", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            # No face detected - suspicious
            is_suspicious = True
        
        return frame, num_faces, is_suspicious
    
    def detect_objects(self, frame):
        """
        Comprehensive object detection using MediaPipe.
        
        Args:
            frame: Input BGR image frame
            
        Returns:
            tuple: (annotated_frame, detection_info, is_suspicious)
        """
        # Detect objects (phones, books, laptops) using EfficientDet
        frame, detected_objects, phone_detected = self.detect_objects_efficientdet(frame)
        
        # Detect faces
        frame, num_faces, faces_suspicious = self.detect_faces(frame)
        
        # Detect hands
        frame, num_hands, hand_positions = self.detect_hands(frame)
        
        # Overall suspicious status
        is_suspicious = faces_suspicious or num_hands > 2 or phone_detected or len(detected_objects) > 0
        
        # Update state
        self.phone_detected = phone_detected
        self.detected_objects = detected_objects
        
        # Prepare detection info
        detection_info = {
            'num_hands': num_hands,
            'num_faces': num_faces,
            'num_objects': len(detected_objects),
            'hand_positions': hand_positions,
            'phone_detected': phone_detected,
            'detected_objects': detected_objects
        }
        
        # Display detection information
        y_offset = 130  # Starting Y position for text
        
        # Faces
        face_color = (0, 0, 255) if faces_suspicious else (0, 255, 0)
        face_status = "SUSPICIOUS" if faces_suspicious else "OK"
        cv2.putText(frame, f"Faces: {num_faces} ({face_status})", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2)
        
        # Hands
        y_offset += 30
        hand_color = (0, 0, 255) if num_hands > 2 else (0, 255, 0)
        cv2.putText(frame, f"Hands: {num_hands}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2)
        
        # Phone/Object detection
        y_offset += 30
        phone_color = (0, 0, 255) if phone_detected else (0, 255, 0)
        phone_status = "DETECTED!" if phone_detected else "None"
        cv2.putText(frame, f"Phone: {phone_status}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, phone_color, 2)
        
        # Other suspicious objects
        if detected_objects:
            y_offset += 30
            objects_str = ", ".join(detected_objects)
            cv2.putText(frame, f"Objects: {objects_str}", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Overall status
        y_offset += 30
        status_color = (0, 0, 255) if is_suspicious else (0, 255, 0)
        status = "SUSPICIOUS" if is_suspicious else "NORMAL"
        cv2.putText(frame, f"Status: {status}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        return frame, detection_info, is_suspicious
    
    def release(self):
        """Release MediaPipe resources."""
        self.hands.close()
        self.face_detection.close()
        if self.object_detector is not None:
            self.object_detector.close()
        print("Object detector resources released.")

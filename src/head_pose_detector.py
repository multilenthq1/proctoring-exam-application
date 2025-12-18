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
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        
        # Thresholds for suspicious head movements (in degrees)
        self.pitch_threshold = 20  # Looking up/down
        self.yaw_threshold = 25    # Looking left/right
        
    def detect_head_pose(self, frame):
        """
        Detect head pose from a video frame.
        
        Args:
            frame: Input BGR image frame
            
        Returns:
            tuple: (annotated_frame, pitch, yaw, roll, is_suspicious)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        is_suspicious = False
        pitch, yaw, roll = 0, 0, 0
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get image dimensions
                img_h, img_w, img_c = frame.shape
                face_3d = []
                face_2d = []
                
                # Extract key facial landmarks for pose estimation
                # Landmarks: nose tip, chin, left eye left corner, right eye right corner, 
                # left mouth corner, right mouth corner
                landmark_indices = [1, 33, 263, 61, 291, 199]
                
                for idx in landmark_indices:
                    lm = face_landmarks.landmark[idx]
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    
                    # 2D coordinates
                    face_2d.append([x, y])
                    
                    # 3D coordinates
                    face_3d.append([x, y, lm.z])
                
                # Convert to numpy arrays
                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)
                
                # Camera matrix (approximate)
                focal_length = 1 * img_w
                cam_matrix = np.array([
                    [focal_length, 0, img_h / 2],
                    [0, focal_length, img_w / 2],
                    [0, 0, 1]
                ])
                
                # Distortion matrix (assuming no lens distortion)
                dist_matrix = np.zeros((4, 1), dtype=np.float64)
                
                # Solve PnP to get rotation and translation vectors
                success, rot_vec, trans_vec = cv2.solvePnP(
                    face_3d, face_2d, cam_matrix, dist_matrix
                )
                
                # Get rotation matrix
                rmat, jac = cv2.Rodrigues(rot_vec)
                
                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
                
                # Get rotation degrees
                pitch = angles[0] * 360
                yaw = angles[1] * 360
                roll = angles[2] * 360
                
                # Check if head pose is suspicious
                if abs(pitch) > self.pitch_threshold or abs(yaw) > self.yaw_threshold:
                    is_suspicious = True
                    color = (0, 0, 255)  # Red
                    status = "SUSPICIOUS"
                else:
                    color = (0, 255, 0)  # Green
                    status = "NORMAL"
                
                # Display head pose information
                cv2.putText(frame, f"Pitch: {int(pitch)}", (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame, f"Yaw: {int(yaw)}", (20, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame, f"Roll: {int(roll)}", (20, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame, f"Head Pose: {status}", (20, 140), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Draw nose direction
                nose_3d = (face_3d[0, 0], face_3d[0, 1], face_3d[0, 2] * 3000)
                nose_2d, _ = cv2.projectPoints(
                    np.array([nose_3d]), rot_vec, trans_vec, cam_matrix, dist_matrix
                )
                
                p1 = (int(face_2d[0, 0]), int(face_2d[0, 1]))
                p2 = (int(nose_2d[0][0][0]), int(nose_2d[0][0][1]))
                
                cv2.line(frame, p1, p2, color, 3)
                
        return frame, pitch, yaw, roll, is_suspicious
    
    def release(self):
        """Release resources."""
        self.face_mesh.close()

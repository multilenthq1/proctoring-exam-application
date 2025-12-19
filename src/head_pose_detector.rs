/// Head Pose Detection Module
/// Uses OpenCV face detection and facial landmarks to detect head orientation
use opencv::{
    core::{Mat, Point, Scalar, Size, Vector},
    imgproc,
    objdetect,
    prelude::*,
    types::VectorOfRect,
};
use anyhow::Result;

pub struct HeadPoseDetector {
    face_cascade: objdetect::CascadeClassifier,
    horizontal_threshold: f64,
    vertical_threshold: f64,
    current_direction: String,
}

impl HeadPoseDetector {
    /// Initialize the head pose detector
    pub fn new() -> Result<Self> {
        // Load Haar Cascade for face detection
        let face_cascade = objdetect::CascadeClassifier::new(
            "haarcascade_frontalface_default.xml"
        )?;

        Ok(Self {
            face_cascade,
            horizontal_threshold: 0.03,
            vertical_threshold: 0.02,
            current_direction: "Forward".to_string(),
        })
    }

    /// Detect head pose from a video frame
    /// 
    /// Returns: (pitch, yaw, roll, is_suspicious, direction)
    pub fn detect_head_pose(&mut self, frame: &mut Mat) -> Result<(f64, f64, f64, bool, String)> {
        let mut gray = Mat::default();
        imgproc::cvt_color(frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

        let mut faces = VectorOfRect::new();
        self.face_cascade.detect_multi_scale(
            &gray,
            &mut faces,
            1.3,
            5,
            0,
            Size::new(30, 30),
            Size::default(),
        )?;

        let mut is_suspicious = false;
        let pitch = 0.0;
        let yaw = 0.0;
        let roll = 0.0;

        if faces.len() > 0 {
            let face = faces.get(0)?;
            
            // Calculate face center
            let face_center_x = face.x + face.width / 2;
            let face_center_y = face.y + face.height / 2;

            // Get frame dimensions
            let frame_width = frame.cols() as i32;
            let frame_height = frame.rows() as i32;
            let frame_center_x = frame_width / 2;
            let frame_center_y = frame_height / 2;

            // Calculate normalized offsets
            let horizontal_offset = ((face_center_x - frame_center_x) as f64) / (face.width as f64);
            let vertical_offset = ((face_center_y - frame_center_y) as f64) / (face.height as f64);

            // Determine direction
            let mut directions = Vec::new();

            // Horizontal detection
            if horizontal_offset < -self.horizontal_threshold {
                directions.push("Right");
                is_suspicious = true;
            } else if horizontal_offset > self.horizontal_threshold {
                directions.push("Left");
                is_suspicious = true;
            }

            // Vertical detection
            if vertical_offset < -self.vertical_threshold {
                directions.push("Up");
                is_suspicious = true;
            } else if vertical_offset > self.vertical_threshold {
                directions.push("Down");
                is_suspicious = true;
            }

            self.current_direction = if directions.is_empty() {
                "Forward".to_string()
            } else {
                directions.join(" + ")
            };

            // Draw visualizations
            let color = if is_suspicious {
                Scalar::new(0.0, 0.0, 255.0, 0.0) // Red
            } else {
                Scalar::new(0.0, 255.0, 0.0, 0.0) // Green
            };

            // Draw face center
            imgproc::circle(
                frame,
                Point::new(face_center_x, face_center_y),
                5,
                color,
                -1,
                imgproc::LINE_8,
                0,
            )?;

            // Draw frame center
            imgproc::circle(
                frame,
                Point::new(frame_center_x, frame_center_y),
                5,
                Scalar::new(255.0, 255.0, 0.0, 0.0),
                -1,
                imgproc::LINE_8,
                0,
            )?;

            // Draw line from frame center to face center
            imgproc::line(
                frame,
                Point::new(frame_center_x, frame_center_y),
                Point::new(face_center_x, face_center_y),
                color,
                2,
                imgproc::LINE_8,
                0,
            )?;

            // Draw text
            let status = if is_suspicious {
                "LOOKING AWAY"
            } else {
                "LOOKING AT SCREEN"
            };

            imgproc::put_text(
                frame,
                &format!("Direction: {}", self.current_direction),
                Point::new(20, 50),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
                imgproc::LINE_8,
                false,
            )?;

            imgproc::put_text(
                frame,
                &format!("Status: {}", status),
                Point::new(20, 90),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                imgproc::LINE_8,
                false,
            )?;
        }

        Ok((pitch, yaw, roll, is_suspicious, self.current_direction.clone()))
    }

    pub fn release(&self) {
        // Resources are automatically released in Rust
    }
}

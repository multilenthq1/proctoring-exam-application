/// Eye Tracking Module
/// Detects and analyzes eye movements using OpenCV
use opencv::{
    core::{Mat, Point, Rect, Scalar, Size, Vector},
    imgproc,
    objdetect,
    prelude::*,
    types::VectorOfRect,
};
use anyhow::Result;

pub struct EyeTracker {
    face_cascade: objdetect::CascadeClassifier,
    eye_cascade: objdetect::CascadeClassifier,
    gaze_threshold: f64,
}

impl EyeTracker {
    /// Initialize the eye tracker
    pub fn new() -> Result<Self> {
        let face_cascade = objdetect::CascadeClassifier::new(
            "haarcascade_frontalface_default.xml"
        )?;

        let eye_cascade = objdetect::CascadeClassifier::new(
            "haarcascade_eye.xml"
        )?;

        Ok(Self {
            face_cascade,
            eye_cascade,
            gaze_threshold: 0.15,
        })
    }

    /// Track eyes and detect suspicious behavior
    /// 
    /// Returns: (gaze_direction, is_suspicious)
    pub fn track_eyes(&self, frame: &mut Mat) -> Result<(String, bool)> {
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

        let mut gaze_direction = "CENTER".to_string();
        let mut is_suspicious = false;

        if faces.len() > 0 {
            let face = faces.get(0)?;
            
            // Create ROI for the face
            let face_roi = Mat::roi(&gray, face)?;
            
            // Detect eyes within the face ROI
            let mut eyes = VectorOfRect::new();
            self.eye_cascade.detect_multi_scale(
                &face_roi,
                &mut eyes,
                1.1,
                10,
                0,
                Size::new(15, 15),
                Size::default(),
            )?;

            // Process detected eyes
            if eyes.len() >= 2 {
                // Get the two largest eye regions (assuming they are the actual eyes)
                let eye1 = eyes.get(0)?;
                let eye2 = eyes.get(1)?;

                // Calculate eye centers relative to face
                let eye1_center_x = (eye1.x + eye1.width / 2) as f64 / face.width as f64;
                let eye2_center_x = (eye2.x + eye2.width / 2) as f64 / face.width as f64;

                let avg_eye_x = (eye1_center_x + eye2_center_x) / 2.0;

                // Determine gaze direction based on eye position
                if avg_eye_x < (0.5 - self.gaze_threshold) {
                    gaze_direction = "LEFT".to_string();
                    is_suspicious = true;
                } else if avg_eye_x > (0.5 + self.gaze_threshold) {
                    gaze_direction = "RIGHT".to_string();
                    is_suspicious = true;
                }

                // Draw eyes on the original frame
                for i in 0..eyes.len() {
                    let eye = eyes.get(i)?;
                    let eye_rect = Rect::new(
                        face.x + eye.x,
                        face.y + eye.y,
                        eye.width,
                        eye.height,
                    );

                    imgproc::rectangle(
                        frame,
                        eye_rect,
                        Scalar::new(0.0, 255.0, 255.0, 0.0),
                        2,
                        imgproc::LINE_8,
                        0,
                    )?;
                }
            } else if eyes.len() < 2 {
                // Not enough eyes detected - suspicious
                gaze_direction = "EYES NOT VISIBLE".to_string();
                is_suspicious = true;
            }

            // Display information
            let color = if is_suspicious {
                Scalar::new(0.0, 0.0, 255.0, 0.0) // Red
            } else {
                Scalar::new(0.0, 255.0, 0.0, 0.0) // Green
            };

            let status = if is_suspicious { "SUSPICIOUS" } else { "NORMAL" };

            imgproc::put_text(
                frame,
                &format!("Gaze: {}", gaze_direction),
                Point::new(20, 170),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                imgproc::LINE_8,
                false,
            )?;

            imgproc::put_text(
                frame,
                &format!("Eye Status: {}", status),
                Point::new(20, 200),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                imgproc::LINE_8,
                false,
            )?;
        }

        Ok((gaze_direction, is_suspicious))
    }

    pub fn release(&self) {
        // Resources are automatically released in Rust
    }
}

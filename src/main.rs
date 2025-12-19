// Proctoring Exam Application - Main Module
// Integrates all proctoring features: head pose detection, noise detection,
// eye tracking, and object detection.

mod head_pose_detector;
mod noise_detector;
mod eye_tracker;
mod object_detector;

use opencv::{
    core::{Mat, Point, Scalar},
    highgui,
    imgproc,
    prelude::*,
    videoio,
};
use chrono::{DateTime, Local};
use std::fs::OpenOptions;
use std::io::Write;
use anyhow::Result;

use head_pose_detector::HeadPoseDetector;
use noise_detector::NoiseDetector;
use eye_tracker::EyeTracker;
use object_detector::ObjectDetector;

pub struct ProctoringExamApp {
    head_pose_detector: HeadPoseDetector,
    noise_detector: NoiseDetector,
    eye_tracker: EyeTracker,
    object_detector: ObjectDetector,
    violations: Vec<String>,
    violation_file: String,
    session_start_time: Option<std::time::Instant>,
    frame_count: usize,
}

impl ProctoringExamApp {
    /// Initialize all proctoring modules
    pub fn new() -> Result<Self> {
        println!("Initializing Proctoring Exam Application...");

        let head_pose_detector = HeadPoseDetector::new()?;
        let noise_detector = NoiseDetector::new(50.0);
        let eye_tracker = EyeTracker::new()?;
        let object_detector = ObjectDetector::new()?;

        println!("Initialization complete!");

        Ok(Self {
            head_pose_detector,
            noise_detector,
            eye_tracker,
            object_detector,
            violations: Vec::new(),
            violation_file: "violations.txt".to_string(),
            session_start_time: None,
            frame_count: 0,
        })
    }

    /// Log a violation
    fn log_violation(&mut self, violation_type: &str, details: &str) {
        let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
        let violation_entry = format!("[{}] {}: {}", timestamp, violation_type, details);

        self.violations.push(violation_entry.clone());

        // Write to file
        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.violation_file)
        {
            writeln!(file, "{}", violation_entry).ok();
        }

        println!("VIOLATION LOGGED: {}", violation_entry);
    }

    /// Process a single frame
    fn process_frame(&mut self, frame: &mut Mat) -> Result<()> {
        self.frame_count += 1;

        // Head Pose Detection
        let (_pitch, _yaw, _roll, head_suspicious, direction) = 
            self.head_pose_detector.detect_head_pose(frame)?;
        
        if head_suspicious {
            self.log_violation("HEAD_POSE", &format!("Direction: {}", direction));
        }

        // Eye Tracking
        let (gaze_direction, eye_suspicious) = self.eye_tracker.track_eyes(frame)?;
        
        if eye_suspicious {
            self.log_violation("EYE_TRACKING", &format!("Gaze Direction: {}", gaze_direction));
        }

        // Object Detection
        let (detection_info, object_suspicious) = self.object_detector.detect_objects(frame)?;
        
        if object_suspicious {
            let mut details = format!(
                "Faces: {}, Objects: {}",
                detection_info.num_faces, detection_info.num_objects
            );
            if detection_info.phone_detected {
                details.push_str(", PHONE DETECTED");
            }
            self.log_violation("OBJECT_DETECTION", &details);
        }

        // Noise Detection
        let (noise_level, noise_suspicious) = self.noise_detector.detect_noise();
        
        if noise_suspicious {
            self.log_violation("NOISE_LEVEL", &format!("Noise: {} dB", noise_level as i32));
        }

        // Display noise level
        let noise_color = if noise_suspicious {
            Scalar::new(0.0, 0.0, 255.0, 0.0)
        } else {
            Scalar::new(0.0, 255.0, 0.0, 0.0)
        };

        imgproc::put_text(
            frame,
            &format!("Noise: {} dB", noise_level as i32),
            Point::new(20, 250),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.6,
            noise_color,
            2,
            imgproc::LINE_8,
            false,
        )?;

        // Display session information
        if let Some(start_time) = self.session_start_time {
            let elapsed = start_time.elapsed().as_secs();
            
            imgproc::put_text(
                frame,
                &format!("Session Time: {}s", elapsed),
                Point::new(20, 290),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.6,
                Scalar::new(255.0, 255.0, 255.0, 0.0),
                2,
                imgproc::LINE_8,
                false,
            )?;

            imgproc::put_text(
                frame,
                &format!("Violations: {}", self.violations.len()),
                Point::new(20, 320),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.6,
                Scalar::new(255.0, 255.0, 0.0, 0.0),
                2,
                imgproc::LINE_8,
                false,
            )?;
        }

        // Display instructions
        let width = frame.cols();
        imgproc::put_text(
            frame,
            "Press 'q' to quit",
            Point::new(width - 200, 30),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.5,
            Scalar::new(255.0, 255.0, 255.0, 0.0),
            1,
            imgproc::LINE_8,
            false,
        )?;

        Ok(())
    }

    /// Main application loop
    pub fn run(&mut self) -> Result<()> {
        println!("\n{}", "=".repeat(60));
        println!("PROCTORING EXAM APPLICATION");
        println!("{}", "=".repeat(60));
        println!("\nStarting exam session...");
        println!("Please ensure:");
        println!("  - You are in a well-lit room");
        println!("  - Your face is clearly visible");
        println!("  - No unauthorized materials are present");
        println!("\nPress 'q' to quit the application");
        println!("{}\n", "=".repeat(60));

        // Start camera
        let mut cap = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
        
        if !videoio::VideoCapture::is_opened(&cap)? {
            anyhow::bail!("Failed to open camera");
        }

        // Set camera properties
        cap.set(videoio::CAP_PROP_FRAME_WIDTH, 1280.0)?;
        cap.set(videoio::CAP_PROP_FRAME_HEIGHT, 720.0)?;
        cap.set(videoio::CAP_PROP_FPS, 30.0)?;

        println!("Camera started successfully");

        // Start noise monitoring
        self.noise_detector.start_monitoring()?;

        // Initialize session
        self.session_start_time = Some(std::time::Instant::now());

        // Clear previous violations file
        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.violation_file)
        {
            writeln!(
                file,
                "Exam Session Started: {}",
                Local::now().format("%Y-%m-%d %H:%M:%S")
            )
            .ok();
            writeln!(file, "{}\n", "=".repeat(60)).ok();
        }

        // Create window
        let window_name = "Proctoring Exam Application";
        highgui::named_window(window_name, highgui::WINDOW_AUTOSIZE)?;

        // Main processing loop
        loop {
            let mut frame = Mat::default();
            cap.read(&mut frame)?;

            if frame.empty() {
                println!("Error: Failed to capture frame");
                break;
            }

            // Process frame
            self.process_frame(&mut frame)?;

            // Display frame
            highgui::imshow(window_name, &frame)?;

            // Check for quit key
            let key = highgui::wait_key(1)?;
            if key == 'q' as i32 {
                println!("\nExiting application...");
                break;
            }
        }

        // Cleanup
        self.cleanup();

        Ok(())
    }

    /// Cleanup resources and save session summary
    fn cleanup(&mut self) {
        println!("\nCleaning up resources...");

        // Stop noise monitoring
        self.noise_detector.stop_monitoring();

        // Release detectors
        self.head_pose_detector.release();
        self.eye_tracker.release();
        self.object_detector.release();

        // Close windows
        highgui::destroy_all_windows().ok();

        // Generate session summary
        let session_duration = if let Some(start_time) = self.session_start_time {
            start_time.elapsed().as_secs()
        } else {
            0
        };

        println!("\n{}", "=".repeat(60));
        println!("EXAM SESSION SUMMARY");
        println!("{}", "=".repeat(60));
        println!("Session Duration: {} seconds", session_duration);
        println!("Total Frames Processed: {}", self.frame_count);
        println!("Total Violations: {}", self.violations.len());
        println!("Violations logged to: {}", self.violation_file);
        println!("{}\n", "=".repeat(60));

        // Write summary to file
        if let Ok(mut file) = OpenOptions::new()
            .append(true)
            .open(&self.violation_file)
        {
            writeln!(file, "\n{}", "=".repeat(60)).ok();
            writeln!(file, "SESSION SUMMARY").ok();
            writeln!(file, "{}", "=".repeat(60)).ok();
            writeln!(file, "Session Duration: {} seconds", session_duration).ok();
            writeln!(file, "Total Frames Processed: {}", self.frame_count).ok();
            writeln!(file, "Total Violations: {}", self.violations.len()).ok();
            writeln!(
                file,
                "Session Ended: {}",
                Local::now().format("%Y-%m-%d %H:%M:%S")
            )
            .ok();
        }
    }
}

fn main() -> Result<()> {
    let mut app = ProctoringExamApp::new()?;
    app.run()
}

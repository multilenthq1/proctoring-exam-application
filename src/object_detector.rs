/// Object Detection Module
/// Detects unauthorized objects and multiple faces
use opencv::{
    core::{Mat, Point, Rect, Scalar, Size, Vector},
    dnn::{self, Net},
    imgproc,
    objdetect,
    prelude::*,
    types::{VectorOfRect, VectorOfString},
};
use anyhow::Result;
use std::collections::HashMap;

pub struct DetectionInfo {
    pub num_hands: usize,
    pub num_faces: usize,
    pub num_objects: usize,
    pub phone_detected: bool,
    pub detected_objects: Vec<String>,
}

pub struct ObjectDetector {
    face_cascade: objdetect::CascadeClassifier,
    dnn_net: Option<Net>,
    class_names: Vec<String>,
}

impl ObjectDetector {
    /// Initialize object detector
    pub fn new() -> Result<Self> {
        let face_cascade = objdetect::CascadeClassifier::new(
            "haarcascade_frontalface_default.xml"
        )?;

        // Try to load YOLO model for object detection
        let (dnn_net, class_names) = Self::load_yolo_model().unwrap_or_else(|e| {
            eprintln!("Could not load YOLO model: {}. Object detection will be limited.", e);
            (None, Vec::new())
        });

        println!("Object Detector initialized!");
        Ok(Self {
            face_cascade,
            dnn_net,
            class_names,
        })
    }

    fn load_yolo_model() -> Result<(Option<Net>, Vec<String>)> {
        // Load YOLO tiny model
        let config_path = "yolov3-tiny.cfg";
        let weights_path = "yolov3-tiny.weights";

        // Check if files exist
        if !std::path::Path::new(config_path).exists() || !std::path::Path::new(weights_path).exists() {
            return Ok((None, Vec::new()));
        }

        let net = dnn::read_net_from_darknet(config_path, weights_path)?;
        
        // COCO class names
        let class_names = vec![
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ].iter().map(|&s| s.to_string()).collect();

        println!("YOLO model loaded successfully!");
        Ok((Some(net), class_names))
    }

    /// Detect faces in the frame
    pub fn detect_faces(&self, frame: &mut Mat) -> Result<(usize, bool)> {
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

        let num_faces = faces.len();
        let is_suspicious = num_faces != 1;

        // Draw bounding boxes
        for i in 0..num_faces {
            let face = faces.get(i)?;
            let color = if is_suspicious {
                Scalar::new(0.0, 0.0, 255.0, 0.0) // Red
            } else {
                Scalar::new(0.0, 255.0, 0.0, 0.0) // Green
            };

            imgproc::rectangle(
                frame,
                face,
                color,
                2,
                imgproc::LINE_8,
                0,
            )?;

            imgproc::put_text(
                frame,
                "Face",
                Point::new(face.x, face.y - 10),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                imgproc::LINE_8,
                false,
            )?;
        }

        Ok((num_faces, is_suspicious))
    }

    /// Detect objects using YOLO
    pub fn detect_objects_yolo(&mut self, frame: &Mat) -> Result<(Vec<String>, bool)> {
        let mut detected_objects = Vec::new();
        let mut phone_detected = false;

        if let Some(ref mut net) = self.dnn_net {
            // Create blob from frame
            let blob = dnn::blob_from_image(
                frame,
                1.0 / 255.0,
                Size::new(416, 416),
                Scalar::new(0.0, 0.0, 0.0, 0.0),
                true,
                false,
                opencv::core::CV_32F,
            )?;

            net.set_input(&blob, "", 1.0, Scalar::default())?;

            // Get output layer names
            let mut output_names = VectorOfString::new();
            let layer_names = net.get_unconnected_out_layers_names()?;
            for i in 0..layer_names.len() {
                output_names.push(layer_names.get(i)?);
            }

            // Forward pass
            let mut outputs = Vector::<Mat>::new();
            net.forward(&mut outputs, &output_names)?;

            // Process outputs
            for i in 0..outputs.len() {
                let output = outputs.get(i)?;
                // Process detections here
                // This is simplified - full implementation would process YOLO output format
            }
        }

        Ok((detected_objects, phone_detected))
    }

    /// Comprehensive object detection
    pub fn detect_objects(&mut self, frame: &mut Mat) -> Result<(DetectionInfo, bool)> {
        // Detect faces
        let (num_faces, faces_suspicious) = self.detect_faces(frame)?;

        // Detect objects using YOLO
        let (detected_objects, phone_detected) = self.detect_objects_yolo(frame)?;

        // For now, we'll use a simplified hand detection (can be enhanced)
        let num_hands = 0;

        let is_suspicious = faces_suspicious || phone_detected || !detected_objects.is_empty();

        let detection_info = DetectionInfo {
            num_hands,
            num_faces,
            num_objects: detected_objects.len(),
            phone_detected,
            detected_objects: detected_objects.clone(),
        };

        // Display detection information
        let y_offset = 130;

        // Faces
        let face_color = if faces_suspicious {
            Scalar::new(0.0, 0.0, 255.0, 0.0) // Red
        } else {
            Scalar::new(0.0, 255.0, 0.0, 0.0) // Green
        };
        let face_status = if faces_suspicious { "SUSPICIOUS" } else { "OK" };

        imgproc::put_text(
            frame,
            &format!("Faces: {} ({})", num_faces, face_status),
            Point::new(20, y_offset),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.6,
            face_color,
            2,
            imgproc::LINE_8,
            false,
        )?;

        // Hands
        let hand_color = if num_hands > 2 {
            Scalar::new(0.0, 0.0, 255.0, 0.0)
        } else {
            Scalar::new(0.0, 255.0, 0.0, 0.0)
        };

        imgproc::put_text(
            frame,
            &format!("Hands: {}", num_hands),
            Point::new(20, y_offset + 30),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.6,
            hand_color,
            2,
            imgproc::LINE_8,
            false,
        )?;

        // Phone detection
        let phone_color = if phone_detected {
            Scalar::new(0.0, 0.0, 255.0, 0.0)
        } else {
            Scalar::new(0.0, 255.0, 0.0, 0.0)
        };
        let phone_status = if phone_detected { "DETECTED!" } else { "None" };

        imgproc::put_text(
            frame,
            &format!("Phone: {}", phone_status),
            Point::new(20, y_offset + 60),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.6,
            phone_color,
            2,
            imgproc::LINE_8,
            false,
        )?;

        // Overall status
        let status_color = if is_suspicious {
            Scalar::new(0.0, 0.0, 255.0, 0.0)
        } else {
            Scalar::new(0.0, 255.0, 0.0, 0.0)
        };
        let status = if is_suspicious { "SUSPICIOUS" } else { "NORMAL" };

        imgproc::put_text(
            frame,
            &format!("Status: {}", status),
            Point::new(20, y_offset + 90),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2,
            imgproc::LINE_8,
            false,
        )?;

        Ok((detection_info, is_suspicious))
    }

    pub fn release(&self) {
        // Resources are automatically released in Rust
    }
}

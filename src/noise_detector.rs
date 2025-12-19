/// Noise Detection Module
/// Monitors ambient noise levels using audio input
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Stream, StreamConfig};
use std::sync::{Arc, Mutex};
use anyhow::Result;

pub struct NoiseDetector {
    threshold_db: f32,
    noise_level: Arc<Mutex<f32>>,
    _stream: Option<Stream>,
    noise_history: Arc<Mutex<Vec<f32>>>,
    history_size: usize,
}

impl NoiseDetector {
    /// Initialize noise detector
    pub fn new(threshold_db: f32) -> Self {
        Self {
            threshold_db,
            noise_level: Arc::new(Mutex::new(0.0)),
            _stream: None,
            noise_history: Arc::new(Mutex::new(Vec::new())),
            history_size: 10,
        }
    }

    /// Calculate RMS (Root Mean Square) of audio samples
    fn calculate_rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        let sum_squares: f32 = samples.iter().map(|&s| s * s).sum();
        (sum_squares / samples.len() as f32).sqrt()
    }

    /// Convert RMS to decibels
    fn rms_to_db(rms: f32) -> f32 {
        if rms == 0.0 {
            return 0.0;
        }

        // Reference value for decibel calculation
        let db = 20.0 * (rms / 1.0).log10() + 90.0;
        db.max(0.0)
    }

    /// Start monitoring audio
    pub fn start_monitoring(&mut self) -> Result<()> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| anyhow::anyhow!("No input device available"))?;

        let config = device.default_input_config()?;

        let noise_level = Arc::clone(&self.noise_level);
        let noise_history = Arc::clone(&self.noise_history);
        let history_size = self.history_size;
        let threshold_db = self.threshold_db;

        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => Self::build_stream::<f32>(
                &device,
                &config.into(),
                noise_level,
                noise_history,
                history_size,
            )?,
            cpal::SampleFormat::I16 => Self::build_stream::<i16>(
                &device,
                &config.into(),
                noise_level,
                noise_history,
                history_size,
            )?,
            cpal::SampleFormat::U16 => Self::build_stream::<u16>(
                &device,
                &config.into(),
                noise_level,
                noise_history,
                history_size,
            )?,
            _ => return Err(anyhow::anyhow!("Unsupported sample format")),
        };

        stream.play()?;
        self._stream = Some(stream);

        println!("Noise monitoring started...");
        Ok(())
    }

    fn build_stream<T>(
        device: &Device,
        config: &StreamConfig,
        noise_level: Arc<Mutex<f32>>,
        noise_history: Arc<Mutex<Vec<f32>>>,
        history_size: usize,
    ) -> Result<Stream>
    where
        T: cpal::Sample,
    {
        let stream = device.build_input_stream(
            config,
            move |data: &[T], _: &cpal::InputCallbackInfo| {
                // Convert samples to f32
                let samples: Vec<f32> = data
                    .iter()
                    .map(|&s| s.to_sample::<f32>())
                    .collect();

                // Calculate RMS and convert to dB
                let rms = Self::calculate_rms(&samples);
                let db = Self::rms_to_db(rms);

                // Update noise level
                if let Ok(mut level) = noise_level.lock() {
                    *level = db;
                }

                // Update history
                if let Ok(mut history) = noise_history.lock() {
                    history.push(db);
                    if history.len() > history_size {
                        history.remove(0);
                    }
                }
            },
            |err| eprintln!("Audio stream error: {}", err),
            None,
        )?;

        Ok(stream)
    }

    /// Detect current noise level
    /// 
    /// Returns: (noise_level_db, is_suspicious)
    pub fn detect_noise(&self) -> (f32, bool) {
        // Calculate average from history
        let avg_noise = if let Ok(history) = self.noise_history.lock() {
            if history.is_empty() {
                0.0
            } else {
                history.iter().sum::<f32>() / history.len() as f32
            }
        } else {
            0.0
        };

        let is_suspicious = avg_noise > self.threshold_db;
        (avg_noise, is_suspicious)
    }

    /// Stop monitoring
    pub fn stop_monitoring(&mut self) {
        if let Some(stream) = self._stream.take() {
            drop(stream);
        }
        println!("Noise monitoring stopped.");
    }
}

impl Drop for NoiseDetector {
    fn drop(&mut self) {
        self.stop_monitoring();
    }
}

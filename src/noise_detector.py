"""
Noise Detection Module
Monitors ambient noise levels to detect unauthorized voices or sounds.
"""

import numpy as np
import pyaudio
import struct
import math


class NoiseDetector:
    """
    Detects and monitors ambient noise levels using PyAudio.
    Identifies suspicious noise levels that may indicate cheating.
    """
    
    def __init__(self, threshold_db=50, chunk_size=1024, sample_rate=44100):
        """
        Initialize noise detector.
        
        Args:
            threshold_db: Noise threshold in decibels (default: 50 dB)
            chunk_size: Number of audio frames per buffer (default: 1024)
            sample_rate: Audio sample rate in Hz (default: 44100)
        """
        self.threshold_db = threshold_db
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.format = pyaudio.paInt16
        self.channels = 1
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # Noise level history for smoothing
        self.noise_history = []
        self.history_size = 10
        
    def start_monitoring(self):
        """Start the audio stream for noise monitoring."""
        try:
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            print("Noise monitoring started...")
        except Exception as e:
            print(f"Error starting audio stream: {e}")
            print("Continuing without noise detection...")
    
    def calculate_rms(self, data):
        """
        Calculate Root Mean Square (RMS) of audio data.
        
        Args:
            data: Raw audio data bytes
            
        Returns:
            float: RMS value
        """
        count = len(data) // 2  # Use integer division
        format_str = "%dh" % count
        shorts = struct.unpack(format_str, data)
        
        sum_squares = sum(s ** 2 for s in shorts)
        rms = math.sqrt(sum_squares / count)
        
        return rms
    
    def rms_to_db(self, rms):
        """
        Convert RMS value to decibels.
        
        Args:
            rms: RMS value
            
        Returns:
            float: Decibel value
        """
        if rms == 0:
            return 0
        
        # Reference value for decibel calculation
        db = 20 * math.log10(rms / 32768.0) + 90
        return max(0, db)
    
    def detect_noise(self):
        """
        Detect current noise level.
        
        Returns:
            tuple: (noise_level_db, is_suspicious)
        """
        if self.stream is None or not self.stream.is_active():
            return 0, False
        
        try:
            # Read audio data
            data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            
            # Calculate RMS and convert to dB
            rms = self.calculate_rms(data)
            noise_db = self.rms_to_db(rms)
            
            # Add to history for smoothing
            self.noise_history.append(noise_db)
            if len(self.noise_history) > self.history_size:
                self.noise_history.pop(0)
            
            # Calculate average noise level
            avg_noise = sum(self.noise_history) / len(self.noise_history)
            
            # Check if noise is suspicious
            is_suspicious = avg_noise > self.threshold_db
            
            return avg_noise, is_suspicious
            
        except Exception as e:
            print(f"Error reading audio: {e}")
            return 0, False
    
    def stop_monitoring(self):
        """Stop audio stream and release resources."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        print("Noise monitoring stopped.")

import numpy as np
from scipy import signal
import librosa
import noisereduce as nr
from typing import Tuple, Dict, Any


class AudioCleaner:
    """
    Professional audio cleaning module for speech enhancement.
    
    Features:
    - Noise reduction using spectral subtraction
    - Voice enhancement with high-pass filtering
    - Audio normalization
    - Silence trimming
    - Audio quality analysis
    """
    
    def __init__(self, sr: int = 16000):
        """
        Initialize audio cleaner.
        
        Args:
            sr (int): Sample rate in Hz (default: 16000)
        """
        self.sr = sr
    
    def clean_audio(
        self,
        audio: np.ndarray,
        apply_noise_reduction: bool = True,
        apply_enhancement: bool = True,
        apply_normalization: bool = True,
        apply_trim_silence: bool = True,
        reduction_strength: float = 0.5
    ) -> np.ndarray:
        """
        Apply complete audio cleaning pipeline.
        
        Args:
            audio (np.ndarray): Audio signal
            apply_noise_reduction (bool): Apply noise reduction
            apply_enhancement (bool): Apply voice enhancement
            apply_normalization (bool): Apply normalization
            apply_trim_silence (bool): Trim silence from edges
            reduction_strength (float): Noise reduction strength (0.0-1.0)
        
        Returns:
            np.ndarray: Cleaned audio signal
        """
        cleaned = audio.copy()
        
        if apply_noise_reduction:
            cleaned = self._reduce_noise(cleaned, reduction_strength)
        
        if apply_enhancement:
            cleaned = self._enhance_voice(cleaned)
        
        if apply_normalization:
            cleaned = self._normalize_audio(cleaned)
        
        if apply_trim_silence:
            cleaned = self._trim_silence(cleaned)
        
        return cleaned
    
    def _reduce_noise(
        self,
        audio: np.ndarray,
        reduction_strength: float = 0.5
    ) -> np.ndarray:
        """
        Reduce background noise using spectral subtraction.
        
        Args:
            audio (np.ndarray): Audio signal
            reduction_strength (float): Reduction strength (0.0-1.0)
        
        Returns:
            np.ndarray: Noise-reduced audio
        """
        try:
            # Normalize input strength to noisereduce's prop_decrease parameter
            # prop_decrease ranges from 0.0 to ~1.0, higher = more reduction
            prop_decrease = min(max(reduction_strength, 0.0), 1.0)
            
            reduced_noise = nr.reduce_noise(
                y=audio,
                sr=self.sr,
                prop_decrease=prop_decrease,
                stationary=False
            )
            return reduced_noise
        except Exception as e:
            print(f"Warning: Noise reduction failed ({e}), returning original audio")
            return audio
    
    def _enhance_voice(self, audio: np.ndarray) -> np.ndarray:
        """
        Enhance voice using high-pass filter and compression.
        
        Args:
            audio (np.ndarray): Audio signal
        
        Returns:
            np.ndarray: Enhanced audio
        """
        try:
            # High-pass filter to remove low-frequency rumble
            sos = signal.butter(5, 80, 'hp', fs=self.sr, output='sos')
            filtered = signal.sosfilt(sos, audio)
            
            # Dynamic range compression
            compressed = self._compress_audio(filtered)
            
            return compressed
        except Exception as e:
            print(f"Warning: Voice enhancement failed ({e}), returning original audio")
            return audio
    
    def _compress_audio(self, audio: np.ndarray, ratio: float = 4.0, threshold: float = -20.0) -> np.ndarray:
        """
        Apply dynamic range compression to balance audio levels.
        
        Args:
            audio (np.ndarray): Audio signal
            ratio (float): Compression ratio
            threshold (float): Threshold in dB
        
        Returns:
            np.ndarray: Compressed audio
        """
        # Convert threshold from dB to linear
        threshold_linear = 10 ** (threshold / 20.0)
        
        # Calculate envelope
        abs_audio = np.abs(audio)
        envelope = np.maximum.accumulate(abs_audio)
        
        # Apply compression where signal exceeds threshold
        gain = np.ones_like(audio)
        mask = envelope > threshold_linear
        gain[mask] = (threshold_linear + (envelope[mask] - threshold_linear) / ratio) / np.maximum(envelope[mask], 1e-10)
        
        return audio * gain
    
    def _normalize_audio(self, audio: np.ndarray, target_loudness: float = -20.0) -> np.ndarray:
        """
        Normalize audio to target loudness (LUFS approximation).
        
        Args:
            audio (np.ndarray): Audio signal
            target_loudness (float): Target loudness in dB
        
        Returns:
            np.ndarray: Normalized audio
        """
        try:
            # Estimate loudness using RMS
            rms = np.sqrt(np.mean(audio ** 2))
            
            if rms > 0:
                current_loudness_db = 20 * np.log10(rms + 1e-10)
                gain_db = target_loudness - current_loudness_db
                gain_linear = 10 ** (gain_db / 20.0)
                
                # Apply gain with soft clipping to prevent distortion
                normalized = audio * gain_linear
                normalized = np.tanh(normalized)  # Soft clipping
                
                return normalized
            else:
                return audio
        except Exception as e:
            print(f"Warning: Normalization failed ({e}), returning original audio")
            return audio
    
    def _trim_silence(self, audio: np.ndarray, threshold_db: float = -40.0, min_duration: float = 0.1) -> np.ndarray:
        """
        Trim silence from beginning and end of audio.
        
        Args:
            audio (np.ndarray): Audio signal
            threshold_db (float): Silence threshold in dB
            min_duration (float): Minimum duration to keep (seconds)
        
        Returns:
            np.ndarray: Trimmed audio
        """
        try:
            # Use librosa's silence detection
            threshold_linear = 10 ** (threshold_db / 20.0)
            trimmed, _ = librosa.effects.trim(
                audio,
                top_db=-threshold_db,
                ref=np.max
            )
            return trimmed
        except Exception as e:
            print(f"Warning: Silence trimming failed ({e}), returning original audio")
            return audio
    
    def get_audio_stats(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Calculate comprehensive audio statistics.
        
        Args:
            audio (np.ndarray): Audio signal
        
        Returns:
            dict: Audio quality metrics
        """
        try:
            duration = len(audio) / self.sr
            rms = np.sqrt(np.mean(audio ** 2))
            rms_db = 20 * np.log10(rms + 1e-10)
            peak_amplitude = np.max(np.abs(audio))
            crest_factor = peak_amplitude / (rms + 1e-10)
            
            # Estimate SNR using spectral analysis
            snr = self._estimate_snr(audio)
            
            # Find dominant frequency
            fft = np.fft.fft(audio)
            freqs = np.fft.fftfreq(len(audio), 1/self.sr)
            magnitude = np.abs(fft)
            dominant_freq_idx = np.argmax(magnitude[:len(magnitude)//2])
            dominant_freq = freqs[dominant_freq_idx]
            
            return {
                "duration_seconds": round(duration, 2),
                "rms_db": round(rms_db, 2),
                "peak_amplitude": round(peak_amplitude, 4),
                "crest_factor": round(crest_factor, 2),
                "snr_estimate_db": round(snr, 2),
                "dominant_frequency_hz": round(abs(dominant_freq), 1),
                "sample_rate": self.sr,
                "total_samples": len(audio)
            }
        except Exception as e:
            print(f"Warning: Audio stats calculation failed ({e})")
            return {
                "error": str(e),
                "duration_seconds": len(audio) / self.sr,
                "sample_rate": self.sr,
                "total_samples": len(audio)
            }
    
    def _estimate_snr(self, audio: np.ndarray) -> float:
        """
        Estimate Signal-to-Noise Ratio.
        Uses assumption that noise is in high frequencies, signal in mid-range.
        
        Args:
            audio (np.ndarray): Audio signal
        
        Returns:
            float: SNR in dB
        """
        try:
            # Use STFT for better frequency resolution
            D = librosa.stft(audio)
            magnitude = np.abs(D)
            
            # Assume first 10% of spectrum (low freq) contains mostly signal
            signal_region = magnitude[:magnitude.shape[0]//10, :]
            signal_power = np.mean(signal_region ** 2)
            
            # Assume last 20% of spectrum (high freq) contains mostly noise
            noise_region = magnitude[-magnitude.shape[0]//5:, :]
            noise_power = np.mean(noise_region ** 2)
            
            # Calculate SNR
            snr_db = 10 * np.log10((signal_power + 1e-10) / (noise_power + 1e-10))
            return max(snr_db, 0)  # Return at least 0 dB
        except Exception as e:
            print(f"Warning: SNR estimation failed ({e})")
            return 10.0  # Return default value

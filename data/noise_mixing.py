import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional
import numpy as np
import soundfile as sf

@dataclass
class NoiseConfig:    
    # Coverage parameters
    min_noise_ratio: float = 0.5
    max_noise_ratio: float = 1.0
    
    # SNR parameters
    min_snr_db: float = 0
    max_snr_db: float = 20
    
    # Block duration parameters (in seconds)
    min_block_duration_sec: float = 5.0
    max_block_duration_sec: float = 10.0
    fade_ms: float = 15.0
    
    # Amplitude limits
    max_noise_rms_ratio: float = 0.5
    use_global_speech_rms: bool = True
    
    # Placement parameters
    max_placement_attempts: int = 1000
    target_coverage_tolerance: float = 0.9

class NoiseClipLoader:
    def __init__(
        self,
        noise_paths: list[Path],
        target_sr: int = 16000,
        min_rms_threshold: float = 1e-6,
    ):
        self.noise_paths = noise_paths
        self.target_sr = target_sr
        self.min_rms_threshold = min_rms_threshold

    def _load_clip(self, path: Path) -> Optional[np.ndarray]:
        try:
            data, sr = sf.read(str(path), dtype="float32")

            if sr != self.target_sr:
                return None

            # Convert to mono
            if data.ndim == 2:
                data = data.mean(axis=1, dtype=np.float32)

            # Normalize to exactly 1 second
            target_len = self.target_sr
            if len(data) > target_len:
                data = data[:target_len]
            elif len(data) < target_len:
                data = np.pad(data, (0, target_len - len(data)), mode="constant")

            # Skip near-silent clips
            rms = np.sqrt(np.mean(data**2))
            if rms < self.min_rms_threshold:
                return None

            return data

        except Exception:
            return None

    def get_random_clip(self) -> Optional[np.ndarray]:
        for _ in range(10):
            path = random.choice(self.noise_paths)
            clip = self._load_clip(path)
            if clip is not None:
                return clip
        return None

def apply_fade(audio: np.ndarray, sr: int, fade_ms: float = 30.0) -> np.ndarray:
    """Apply linear fade-in/out to prevent clicks."""
    n = len(audio)
    if n == 0:
        return audio
    
    fade_samples = int(sr * fade_ms / 1000.0)
    fade_samples = np.clip(fade_samples, 1, n // 2)
    
    fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=audio.dtype)
    fade_out = fade_in[::-1]
    
    result = audio.copy()
    result[:fade_samples] *= fade_in
    result[-fade_samples:] *= fade_out
    return result

def calculate_rms(audio: np.ndarray, epsilon: float = 1e-12) -> float:
    """Calculate RMS (root mean square) of audio signal."""
    return float(np.sqrt(np.mean(audio**2) + epsilon))

def scale_noise_to_snr(
    noise: np.ndarray,
    speech_rms: float,
    snr_db: float,
    max_noise_rms_ratio: float,
) -> np.ndarray:
    """Scale noise to achieve target SNR with amplitude cap."""
    noise_rms = calculate_rms(noise)
    
    if noise_rms < 1e-10 or speech_rms < 1e-10:
        return noise * 0.0
    
    target_scale = speech_rms / (noise_rms * (10 ** (snr_db / 20)))
    max_scale = (speech_rms * max_noise_rms_ratio) / noise_rms
    final_scale = min(target_scale, max_scale)
    
    return noise * final_scale

def create_noise_block(
    noise_loader: NoiseClipLoader,
    block_samples: int,
) -> Optional[np.ndarray]:
    noise_1s = noise_loader.get_random_clip()
    if noise_1s is None:
        return None
    
    # Ensure mono
    if noise_1s.ndim == 2:
        noise_1s = noise_1s.mean(axis=1, dtype=np.float32)
    
    # Tile to desired length
    repeats = int(np.ceil(block_samples / len(noise_1s)))
    return np.tile(noise_1s, repeats)[:block_samples]

class IntervalTree:
    """Simple interval tree for efficient overlap detection."""
    
    def __init__(self):
        self.intervals: list[tuple[int, int]] = []
    
    def add(self, start: int, end: int) -> None:
        self.intervals.append((start, end))
    
    def overlaps(self, start: int, end: int) -> bool:
        for s, e in self.intervals:
            if not (end <= s or start >= e):
                return True
        return False

def mix_noise_to_speech(
    speech_wav: np.ndarray,
    noise_paths: list[Path],
    sr: int = 16000,
    config: Optional[NoiseConfig] = None,
) -> np.ndarray:
    
    if config is None:
        config = NoiseConfig()
    
    noise_loader = NoiseClipLoader(noise_paths, sr)

    # Convert to mono float32
    if speech_wav.ndim == 2:
        speech_wav = speech_wav.mean(axis=1)
    speech_wav = speech_wav.astype(np.float32)
    
    n_samples = len(speech_wav)
    
    if n_samples < sr:
        print(f"Warning: Audio too short ({n_samples} samples), returning original")
        return speech_wav
    
    # Calculate speech RMS reference
    global_speech_rms = calculate_rms(speech_wav)
    
    # Determine target coverage
    target_ratio = random.uniform(config.min_noise_ratio, config.max_noise_ratio)
    target_samples = int(n_samples * target_ratio)
    
    # Initialize
    occupied = IntervalTree()
    placed_blocks: list[tuple[int, np.ndarray]] = []
    
    # Estimate attempts needed (using average block size)
    avg_block_duration = (config.min_block_duration_sec + config.max_block_duration_sec) / 2
    avg_block_samples = int(avg_block_duration * sr)
    est_blocks = max(1, int(target_samples / avg_block_samples))
    max_attempts = config.max_placement_attempts
    
    attempts = 0
    total_placed_samples = 0
    failed_loads = 0
    
    while (
        total_placed_samples < target_samples * config.target_coverage_tolerance
        and attempts < max_attempts
    ):
        attempts += 1
        
        # Random block duration for this specific block
        block_duration_sec = random.uniform(
            config.min_block_duration_sec, 
            config.max_block_duration_sec
        )
        block_samples = int(block_duration_sec * sr)
        
        # Ensure block fits in audio
        block_samples = min(block_samples, n_samples)
        if block_samples <= 0:
            break
        
        # Random placement
        max_start = n_samples - block_samples
        if max_start <= 0:
            break
        
        start = random.randint(0, max_start)
        end = start + block_samples
        
        # Check for overlap
        if occupied.overlaps(start, end):
            continue
        
        # Create noise block (lazy load from disk)
        noise_block = create_noise_block(noise_loader, block_samples)
        if noise_block is None:
            failed_loads += 1
            if failed_loads > 20:  # Too many failed loads
                print(f"Warning: {failed_loads} failed noise loads, stopping early")
                break
            continue
        
        # Determine speech RMS reference
        if config.use_global_speech_rms:
            speech_rms = global_speech_rms
        else:
            speech_rms = calculate_rms(speech_wav[start:end])
        
        # Scale noise to target SNR
        snr_db = random.uniform(config.min_snr_db, config.max_snr_db)
        scaled_noise = scale_noise_to_snr(
            noise_block,
            speech_rms,
            snr_db,
            config.max_noise_rms_ratio,
        )
        
        # Apply fade
        scaled_noise = apply_fade(scaled_noise, sr, config.fade_ms)
        
        # Store block
        placed_blocks.append((start, scaled_noise))
        occupied.add(start, end)
        total_placed_samples += block_samples
    
    # Mix noise blocks into speech
    result = speech_wav.copy()
    for start, noise_block in placed_blocks:
        end = start + len(noise_block)
        result[start:end] += noise_block
    
    # Prevent clipping
    result = np.clip(result, -1.0, 1.0)
    
    return result
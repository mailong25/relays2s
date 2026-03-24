import math
import torch
import soundfile as sf
import torch.nn.functional as F
import torchaudio.compliance.kaldi as kaldi
from data.noise_mixing import mix_noise_to_speech

class AudioFeatureExtractor:
    """Extracts mel-scale filterbank features from audio, both offline and in streaming chunks."""

    def __init__(self, chunk_size_seconds=0.16, sampling_rate=16000):
        self.sampling_rate = sampling_rate
        self.mel_bins = 80
        self.frame_shift = 10
        self.frame_length = 25
        
        self.frame_shift_samples = int(self.frame_shift * sampling_rate / 1000)
        self.frame_length_samples = int(self.frame_length * sampling_rate / 1000)
        self.chunk_size_samples = int(chunk_size_seconds * sampling_rate)
        self.chunk_size_frames = int(self.chunk_size_samples // self.frame_shift_samples)

        self.audio_overlap_samples = self.frame_length_samples - self.frame_shift_samples
        self.feature_overlap_frames = 3

        # Buffers for streaming mode
        self._initialize_buffers()

    def _initialize_buffers(self):
        """Initialize buffers for streaming extraction."""
        self.prev_audio_tail = torch.zeros(self.audio_overlap_samples)
        self.prev_feature_tail = torch.zeros(self.feature_overlap_frames, self.mel_bins)

    def reset(self):
        """Reset state for processing a new audio stream."""
        self._initialize_buffers()

    @staticmethod
    def pad_waveform(wav, frame_size):
        """Pad waveform so its length is a multiple of frame_size."""
        pad_amount = math.ceil(wav.shape[0] / frame_size) * frame_size - wav.shape[0]
        return F.pad(wav, (0, pad_amount))

    def pad_to_multiple_chunks(self, wav):
        """Pad waveform so its length is a multiple of the chunk size."""
        return self.pad_waveform(wav, self.chunk_size_samples)

    def _fbank(self, waveform):
        """Wrapper around kaldi.fbank with common settings."""
        return kaldi.fbank(
            waveform=waveform.unsqueeze(0),
            num_mel_bins=self.mel_bins,
            sample_frequency=self.sampling_rate,
            frame_length=self.frame_length,
            frame_shift=self.frame_shift,
            dither=0.0,
        )

    def process_chunk(self, audio_chunk):
        """
        Extract features from a chunk of audio.
        Args:
            audio_chunk: 1D tensor of audio samples (float32, -1..1 range)
        Returns:
            Tensor of shape (frames, mel_bins)
        """
        with torch.no_grad():
            # Scale to int16 range
            normalized_audio = audio_chunk * 32768.0

            # Concatenate previous overlap
            full_audio = torch.cat([self.prev_audio_tail, normalized_audio])

            # Extract features
            mel_features = self._fbank(full_audio).squeeze(0)

            # Prepend feature overlap
            full_features = torch.cat([self.prev_feature_tail, mel_features], dim=0)

            # Update buffers
            self.prev_audio_tail = normalized_audio[-self.audio_overlap_samples:].clone()
            self.prev_feature_tail = mel_features[-self.feature_overlap_frames:, :].clone()

            return full_features

    def extract_offline(self, audio_path, noise_paths = None):
        """
        Extract features from a full audio file (no chunking).
        Args:
            audio_path: Path to audio file
            noise_paths: Whether to augment the audio with noises
        Returns:
            Tensor (frames, mel_bins)
        """
        wav, fs = sf.read(audio_path)
        if noise_paths:
            wav = mix_noise_to_speech(wav, noise_paths, fs)
        wav = torch.tensor(wav, dtype=torch.float32)
        
        # Add overlap padding
        wav = torch.cat([torch.zeros(self.audio_overlap_samples), wav])
        wav = wav * 32768.0
        
        return self._fbank(wav).squeeze(0)

    def extract_streaming(self, audio_path):
        """
        Extract features in streaming chunks.
        Args:
            audio_path: Path to audio file
        Returns:
            Tensor (frames, mel_bins)
        """
        self.reset()

        wav, fs = sf.read(audio_path)
        wav = torch.tensor(wav, dtype=torch.float32)

        # Pad so length is multiple of chunk size
        wav = self.pad_to_multiple_chunks(wav)

        features = []
        for i in range(0, wav.shape[0], self.chunk_size_samples):
            fbank = self.process_chunk(wav[i:i + self.chunk_size_samples])
            features.append(fbank[self.feature_overlap_frames:, :])
        
        return torch.vstack(features)
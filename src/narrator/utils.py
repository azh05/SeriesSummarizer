"""
Utility functions for the narrator module.
"""

import os
import tempfile
from pathlib import Path
from typing import Union, Optional
import torch
import torchaudio


def validate_audio_file(file_path: Union[str, Path]) -> bool:
    """
    Validate if a file is a valid audio file.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        True if the file is a valid audio file, False otherwise
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            return False
            
        # Try to load the audio file
        audio, sample_rate = torchaudio.load(str(file_path))
        return audio.numel() > 0 and sample_rate > 0
        
    except Exception:
        return False


def get_audio_duration(file_path: Union[str, Path]) -> Optional[float]:
    """
    Get the duration of an audio file in seconds.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        Duration in seconds, or None if the file cannot be read
    """
    try:
        audio, sample_rate = torchaudio.load(str(file_path))
        return audio.shape[-1] / sample_rate
    except Exception:
        return None


def create_temp_audio_file(suffix: str = ".wav") -> str:
    """
    Create a temporary audio file path.
    
    Args:
        suffix: File extension (default: ".wav")
        
    Returns:
        Path to the temporary file
    """
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, f"temp_audio{suffix}")
    return temp_file


def ensure_audio_format(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    target_sample_rate: Optional[int] = None,
    target_channels: Optional[int] = None
) -> str:
    """
    Ensure audio file is in the correct format.
    
    Args:
        input_path: Path to input audio file
        output_path: Path for output file. If None, overwrites input.
        target_sample_rate: Target sample rate. If None, keeps original.
        target_channels: Target number of channels. If None, keeps original.
        
    Returns:
        Path to the processed audio file
        
    Raises:
        RuntimeError: If audio processing fails
    """
    try:
        input_path = Path(input_path)
        if output_path is None:
            output_path = input_path
        else:
            output_path = Path(output_path)
            
        # Load audio
        audio, sample_rate = torchaudio.load(str(input_path))
        
        # Resample if needed
        if target_sample_rate and sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=target_sample_rate
            )
            audio = resampler(audio)
            sample_rate = target_sample_rate
            
        # Convert channels if needed
        if target_channels:
            current_channels = audio.shape[0]
            if current_channels != target_channels:
                if target_channels == 1 and current_channels > 1:
                    # Convert to mono by averaging channels
                    audio = audio.mean(dim=0, keepdim=True)
                elif target_channels > 1 and current_channels == 1:
                    # Convert mono to multi-channel by repeating
                    audio = audio.repeat(target_channels, 1)
                else:
                    # For other cases, just take the first N channels or pad with zeros
                    if current_channels > target_channels:
                        audio = audio[:target_channels]
                    else:
                        padding = torch.zeros(
                            target_channels - current_channels,
                            audio.shape[1]
                        )
                        audio = torch.cat([audio, padding], dim=0)
        
        # Save processed audio
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(output_path), audio, sample_rate)
        
        return str(output_path)
        
    except Exception as e:
        raise RuntimeError(f"Failed to process audio file: {str(e)}")


def split_long_text(
    text: str,
    max_length: int = 500,
    split_on_sentences: bool = True
) -> list[str]:
    """
    Split long text into smaller chunks for TTS processing.
    
    Args:
        text: Input text to split
        max_length: Maximum length per chunk
        split_on_sentences: Whether to split on sentence boundaries
        
    Returns:
        List of text chunks
    """
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    
    if split_on_sentences:
        # Split on sentence boundaries
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk + sentence) <= max_length:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
    else:
        # Split at word boundaries
        words = text.split()
        current_chunk = ""
        
        for word in words:
            if len(current_chunk + word + " ") <= max_length:
                current_chunk += word + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = word + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
    
    return chunks


def merge_audio_files(
    input_paths: list[Union[str, Path]],
    output_path: Union[str, Path],
    silence_duration: float = 0.5
) -> str:
    """
    Merge multiple audio files into one.
    
    Args:
        input_paths: List of paths to audio files to merge
        output_path: Path for the merged audio file
        silence_duration: Duration of silence between files (seconds)
        
    Returns:
        Path to the merged audio file
        
    Raises:
        RuntimeError: If merging fails
    """
    try:
        if not input_paths:
            raise ValueError("No input files provided")
        
        # Load first file to get sample rate and format
        first_audio, sample_rate = torchaudio.load(str(input_paths[0]))
        merged_audio = first_audio
        
        # Create silence tensor
        silence_samples = int(silence_duration * sample_rate)
        silence = torch.zeros(first_audio.shape[0], silence_samples)
        
        # Merge all audio files
        for path in input_paths[1:]:
            audio, audio_sr = torchaudio.load(str(path))
            
            # Resample if necessary
            if audio_sr != sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=audio_sr,
                    new_freq=sample_rate
                )
                audio = resampler(audio)
            
            # Match channels
            if audio.shape[0] != merged_audio.shape[0]:
                if audio.shape[0] == 1 and merged_audio.shape[0] > 1:
                    audio = audio.repeat(merged_audio.shape[0], 1)
                elif audio.shape[0] > 1 and merged_audio.shape[0] == 1:
                    merged_audio = merged_audio.repeat(audio.shape[0], 1)
                    silence = torch.zeros(audio.shape[0], silence_samples)
            
            # Concatenate with silence
            merged_audio = torch.cat([merged_audio, silence, audio], dim=1)
        
        # Save merged audio
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(output_path), merged_audio, sample_rate)
        
        return str(output_path)
        
    except Exception as e:
        raise RuntimeError(f"Failed to merge audio files: {str(e)}")


def get_system_info() -> dict:
    """
    Get system information relevant to TTS processing.
    
    Returns:
        Dictionary with system information
    """
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    
    return info

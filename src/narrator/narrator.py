"""
F5-TTS Narrator Interface

This module provides a clean interface for text-to-speech functionality
using the F5-TTS model.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, Union
import torch
import torchaudio
from f5_tts.api import F5TTS


class NarratorInterface:
    """
    A clean interface for F5-TTS text-to-speech functionality.
    
    This class provides methods to convert text to speech using the F5-TTS model,
    with options for different voices and audio configurations.
    """
    
    def __init__(
        self,
        model_type: str = "F5-TTS",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the NarratorInterface.
        
        Args:
            model_type: Type of F5-TTS model to use (default: "F5-TTS")
            device: Device to run the model on (cuda/cpu). Auto-detects if None.
            cache_dir: Directory to cache model files. Uses default if None.
        """
        self.model_type = model_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        self._model = None
        
        print(f"Initializing F5-TTS on device: {self.device}")
    
    def _load_model(self):
        """Lazy load the F5-TTS model."""
        if self._model is None:
            print("Loading F5-TTS model...")
            self._model = F5TTS(
                model_type=self.model_type,
                device=self.device
            )
            print("Model loaded successfully!")
    
    def synthesize(
        self,
        text: str,
        reference_audio: Optional[Union[str, Path]] = None,
        reference_text: Optional[str] = None,
        output_path: Optional[Union[str, Path]] = None,
        speed: float = 1.0,
        remove_silence: bool = True
    ) -> str:
        """
        Convert text to speech using F5-TTS.
        
        Args:
            text: Text to convert to speech
            reference_audio: Path to reference audio file for voice cloning
            reference_text: Text corresponding to the reference audio
            output_path: Path to save the generated audio. If None, saves to temp file.
            speed: Speech speed multiplier (default: 1.0)
            remove_silence: Whether to remove silence from the output
            
        Returns:
            Path to the generated audio file
            
        Raises:
            ValueError: If text is empty or reference audio is provided without reference text
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
            
        if reference_audio and not reference_text:
            raise ValueError("Reference text must be provided when using reference audio")
        
        # Load model if not already loaded
        self._load_model()
        
        # Prepare output path
        if output_path is None:
            temp_dir = tempfile.mkdtemp()
            output_path = os.path.join(temp_dir, "generated_speech.wav")
        else:
            output_path = str(output_path)
            
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"Synthesizing: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        try:
            # Generate audio using F5-TTS
            if reference_audio and reference_text:
                # Voice cloning mode
                print(f"Using reference audio: {reference_audio}")
                audio, sample_rate = self._model.infer(
                    text=text,
                    ref_audio=str(reference_audio),
                    ref_text=reference_text,
                    speed=speed,
                    remove_silence=remove_silence
                )
            else:
                # Default voice mode
                audio, sample_rate = self._model.infer(
                    text=text,
                    speed=speed,
                    remove_silence=remove_silence
                )
            
            # Save the audio
            torchaudio.save(output_path, audio.cpu(), sample_rate)
            print(f"Audio saved to: {output_path}")
            
            return output_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to synthesize audio: {str(e)}")
    
    def synthesize_to_buffer(
        self,
        text: str,
        reference_audio: Optional[Union[str, Path]] = None,
        reference_text: Optional[str] = None,
        speed: float = 1.0,
        remove_silence: bool = True
    ) -> tuple[torch.Tensor, int]:
        """
        Convert text to speech and return audio tensor and sample rate.
        
        Args:
            text: Text to convert to speech
            reference_audio: Path to reference audio file for voice cloning
            reference_text: Text corresponding to the reference audio
            speed: Speech speed multiplier (default: 1.0)
            remove_silence: Whether to remove silence from the output
            
        Returns:
            Tuple of (audio_tensor, sample_rate)
            
        Raises:
            ValueError: If text is empty or reference audio is provided without reference text
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
            
        if reference_audio and not reference_text:
            raise ValueError("Reference text must be provided when using reference audio")
        
        # Load model if not already loaded
        self._load_model()
        
        print(f"Synthesizing to buffer: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        try:
            # Generate audio using F5-TTS
            if reference_audio and reference_text:
                # Voice cloning mode
                print(f"Using reference audio: {reference_audio}")
                audio, sample_rate = self._model.infer(
                    text=text,
                    ref_audio=str(reference_audio),
                    ref_text=reference_text,
                    speed=speed,
                    remove_silence=remove_silence
                )
            else:
                # Default voice mode
                audio, sample_rate = self._model.infer(
                    text=text,
                    speed=speed,
                    remove_silence=remove_silence
                )
            
            return audio, sample_rate
            
        except Exception as e:
            raise RuntimeError(f"Failed to synthesize audio: {str(e)}")
    
    def clone_voice(
        self,
        text: str,
        reference_audio: Union[str, Path],
        reference_text: str,
        output_path: Optional[Union[str, Path]] = None,
        speed: float = 1.0,
        remove_silence: bool = True
    ) -> str:
        """
        Generate speech with voice cloning from a reference audio.
        
        Args:
            text: Text to convert to speech
            reference_audio: Path to reference audio file
            reference_text: Text corresponding to the reference audio
            output_path: Path to save the generated audio. If None, saves to temp file.
            speed: Speech speed multiplier (default: 1.0)
            remove_silence: Whether to remove silence from the output
            
        Returns:
            Path to the generated audio file
        """
        return self.synthesize(
            text=text,
            reference_audio=reference_audio,
            reference_text=reference_text,
            output_path=output_path,
            speed=speed,
            remove_silence=remove_silence
        )
    
    def batch_synthesize(
        self,
        texts: list[str],
        output_dir: Union[str, Path],
        reference_audio: Optional[Union[str, Path]] = None,
        reference_text: Optional[str] = None,
        speed: float = 1.0,
        remove_silence: bool = True,
        filename_prefix: str = "speech"
    ) -> list[str]:
        """
        Synthesize multiple texts to separate audio files.
        
        Args:
            texts: List of texts to convert to speech
            output_dir: Directory to save the generated audio files
            reference_audio: Path to reference audio file for voice cloning
            reference_text: Text corresponding to the reference audio
            speed: Speech speed multiplier (default: 1.0)
            remove_silence: Whether to remove silence from the output
            filename_prefix: Prefix for output filenames
            
        Returns:
            List of paths to the generated audio files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_paths = []
        
        for i, text in enumerate(texts):
            output_path = output_dir / f"{filename_prefix}_{i+1:03d}.wav"
            
            try:
                result_path = self.synthesize(
                    text=text,
                    reference_audio=reference_audio,
                    reference_text=reference_text,
                    output_path=output_path,
                    speed=speed,
                    remove_silence=remove_silence
                )
                output_paths.append(result_path)
                print(f"Completed {i+1}/{len(texts)}: {output_path}")
                
            except Exception as e:
                print(f"Failed to synthesize text {i+1}: {str(e)}")
                continue
        
        return output_paths
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_type": self.model_type,
            "device": self.device,
            "cache_dir": self.cache_dir,
            "model_loaded": self._model is not None
        }
    
    def clear_cache(self):
        """Clear the model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Model cache cleared")

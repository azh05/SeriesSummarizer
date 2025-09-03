#!/usr/bin/env python3
"""
F5-TTS Wrapper Script

A Python wrapper for the f5-tts_infer-cli command that provides a clean interface
for text-to-speech generation with customizable parameters.
"""

import subprocess
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Optional


class F5TTSWrapper:
    """Wrapper class for F5-TTS inference CLI."""
    
    # Default values as specified
    DEFAULT_REF_AUDIO = "audio-sample/jimdale-trimmed.mp3"
    DEFAULT_REF_TEXT = (
        "I read the first Harry Potter Book, and I was blown away. I thought it was "
        "one of the greatest children's books I'd read, and I realized there were "
        "going to be quite a few more of them, and what better to be the narrator. "
        "But having never"
    )
    DEFAULT_MODEL = "F5TTS_v1_Base"
    
    def __init__(self, model: str = DEFAULT_MODEL):
        """
        Initialize the F5-TTS wrapper.
        
        Args:
            model: The F5-TTS model to use (default: F5TTS_v1_Base)
        """
        self.model = model
    
    def generate_speech(
        self,
        gen_text: str,
        output_path: str,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        temp_dir: Optional[str] = None,
        # F5-TTS CLI optional arguments
        config: Optional[str] = None,
        model_cfg: Optional[str] = None,
        ckpt_file: Optional[str] = None,
        vocab_file: Optional[str] = None,
        gen_file: Optional[str] = None,
        output_file: Optional[str] = None,
        save_chunk: bool = False,
        no_legacy_text: bool = False,
        remove_silence: bool = False,
        load_vocoder_from_local: bool = False,
        vocoder_name: Optional[str] = None,
        target_rms: Optional[float] = None,
        cross_fade_duration: Optional[float] = None,
        nfe_step: Optional[int] = None,
        cfg_strength: Optional[float] = None,
        sway_sampling_coef: Optional[float] = None,
        speed: Optional[float] = None,
        fix_duration: Optional[float] = None,
        device: Optional[str] = None
    ) -> bool:
        """
        Generate speech using F5-TTS.
        
        Args:
            gen_text: The text to generate speech for
            output_path: Path where the final audio file should be saved
            ref_audio: Path to reference audio file (uses default if None)
            ref_text: Reference text (uses default if None)
            temp_dir: Temporary directory for CLI output (auto-generated if None)
            config: Configuration file path
            model_cfg: Path to F5-TTS model config file (.yaml)
            ckpt_file: Path to model checkpoint (.pt)
            vocab_file: Path to vocab file (.txt)
            gen_file: File with text to generate (ignores gen_text if provided)
            output_file: Name of output file
            save_chunk: Save each audio chunk during inference
            no_legacy_text: Don't use lossy ASCII transliterations
            remove_silence: Remove long silence from output
            load_vocoder_from_local: Load vocoder from local directory
            vocoder_name: Vocoder to use ('vocos' or 'bigvgan')
            target_rms: Target output speech loudness (default: 0.1)
            cross_fade_duration: Cross-fade duration in seconds (default: 0.15)
            nfe_step: Number of denoising steps (default: 32)
            cfg_strength: Classifier-free guidance strength (default: 2.0)
            sway_sampling_coef: Sway sampling coefficient (default: -1.0)
            speed: Speed of generated audio (default: 1.0)
            fix_duration: Fix total duration in seconds
            device: Device to run on
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Use defaults if not provided
        if ref_audio is None:
            ref_audio = self.DEFAULT_REF_AUDIO
        if ref_text is None:
            ref_text = self.DEFAULT_REF_TEXT
            
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create unique temp directory for CLI output
        if temp_dir is None:
            # Create a unique temp directory name using UUID
            unique_id = str(uuid.uuid4())[:8]  # Use first 8 chars of UUID
            temp_dir = f"temp_f5_output_{unique_id}"
        
        temp_path = Path(temp_dir)
        temp_path.mkdir(exist_ok=True)
        
        try:
            # Build the f5-tts_infer-cli command as a list (proper way to handle quotes)
            cmd = ["f5-tts_infer-cli"]
            
            # Add required arguments
            cmd.extend(["--model", self.model])
            cmd.extend(["--ref_audio", ref_audio])
            cmd.extend(["--ref_text", ref_text])
            
            # Add gen_text or gen_file (gen_file takes precedence)
            if gen_file:
                cmd.extend(["--gen_file", gen_file])
            else:
                cmd.extend(["--gen_text", gen_text])
            
            cmd.extend(["--output_dir", str(temp_path)])
            
            # Add optional arguments if provided
            if config:
                cmd.extend(["--config", config])
            if model_cfg:
                cmd.extend(["--model_cfg", model_cfg])
            if ckpt_file:
                cmd.extend(["--ckpt_file", ckpt_file])
            if vocab_file:
                cmd.extend(["--vocab_file", vocab_file])
            if output_file:
                cmd.extend(["--output_file", output_file])
            if save_chunk:
                cmd.append("--save_chunk")
            if no_legacy_text:
                cmd.append("--no_legacy_text")
            if remove_silence:
                cmd.append("--remove_silence")
            if load_vocoder_from_local:
                cmd.append("--load_vocoder_from_local")
            if vocoder_name:
                cmd.extend(["--vocoder_name", vocoder_name])
            if target_rms is not None:
                cmd.extend(["--target_rms", str(target_rms)])
            if cross_fade_duration is not None:
                cmd.extend(["--cross_fade_duration", str(cross_fade_duration)])
            if nfe_step is not None:
                cmd.extend(["--nfe_step", str(nfe_step)])
            if cfg_strength is not None:
                cmd.extend(["--cfg_strength", str(cfg_strength)])
            if sway_sampling_coef is not None:
                cmd.extend(["--sway_sampling_coef", str(sway_sampling_coef)])
            if speed is not None:
                cmd.extend(["--speed", str(speed)])
            if fix_duration is not None:
                cmd.extend(["--fix_duration", str(fix_duration)])
            if device:
                cmd.extend(["--device", device])
            
            print(f"Running F5-TTS inference...")
            # For display purposes, show a properly quoted version
            display_cmd = []
            for i, arg in enumerate(cmd):
                if i > 0 and not arg.startswith('--'):
                    # This is a parameter value, quote it if it contains spaces or special chars
                    if ' ' in arg or '"' in arg or "'" in arg:
                        # Escape any existing quotes and wrap in quotes
                        escaped_arg = arg.replace('"', '\\"')
                        display_cmd.append(f'"{escaped_arg}"')
                    else:
                        display_cmd.append(arg)
                else:
                    display_cmd.append(arg)
            print(f"Command: {' '.join(display_cmd)}")
            
            # Execute the command (subprocess.run with list handles quoting automatically)
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # The CLI saves to {output_dir}/infer_cli_basic.wav
            source_file = temp_path / "infer_cli_basic.wav"
            
            if not source_file.exists():
                print(f"Error: Expected output file {source_file} was not created")
                return False
            
            # Move the file to the desired output path
            print(f"Moving output from {source_file} to {output_path}")
            shutil.move(str(source_file), output_path)
            
            # Clean up temp directory
            if temp_path.exists() and temp_path.is_dir():
                shutil.rmtree(temp_path)
            
            print(f"Speech generation completed successfully!")
            print(f"Output saved to: {output_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Error running f5-tts_infer-cli: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            return False
        except Exception as e:
            print(f"Unexpected error: {e}")
            return False
        finally:
            # Clean up temp directory if it still exists
            if temp_path.exists() and temp_path.is_dir():
                try:
                    shutil.rmtree(temp_path)
                except Exception as cleanup_error:
                    print(f"Warning: Could not clean up temp directory: {cleanup_error}")


def generate_speech(
    gen_text: str,
    output_path: str,
    ref_audio: Optional[str] = None,
    ref_text: Optional[str] = None,
    model: str = F5TTSWrapper.DEFAULT_MODEL,
    temp_dir: Optional[str] = None,
    # F5-TTS CLI optional arguments
    config: Optional[str] = None,
    model_cfg: Optional[str] = None,
    ckpt_file: Optional[str] = None,
    vocab_file: Optional[str] = None,
    gen_file: Optional[str] = None,
    output_file: Optional[str] = None,
    save_chunk: bool = False,
    no_legacy_text: bool = False,
    remove_silence: bool = False,
    load_vocoder_from_local: bool = False,
    vocoder_name: Optional[str] = None,
    target_rms: Optional[float] = None,
    cross_fade_duration: Optional[float] = None,
    nfe_step: Optional[int] = None,
    cfg_strength: Optional[float] = None,
    sway_sampling_coef: Optional[float] = None,
    speed: Optional[float] = None,
    fix_duration: Optional[float] = None,
    device: Optional[str] = None
) -> bool:
    """
    Convenience function for generating speech with F5-TTS.
    
    Args:
        gen_text: The text to generate speech for
        output_path: Path where the final audio file should be saved
        ref_audio: Path to reference audio file (uses default if None)
        ref_text: Reference text (uses default if None)
        model: The F5-TTS model to use
        temp_dir: Temporary directory for CLI output (auto-generated if None)
        
        All other arguments correspond to F5-TTS CLI options - see class method for details.
        
    Returns:
        bool: True if successful, False otherwise
    """
    wrapper = F5TTSWrapper(model=model)
    return wrapper.generate_speech(
        gen_text=gen_text,
        output_path=output_path,
        ref_audio=ref_audio,
        ref_text=ref_text,
        temp_dir=temp_dir,
        config=config,
        model_cfg=model_cfg,
        ckpt_file=ckpt_file,
        vocab_file=vocab_file,
        gen_file=gen_file,
        output_file=output_file,
        save_chunk=save_chunk,
        no_legacy_text=no_legacy_text,
        remove_silence=remove_silence,
        load_vocoder_from_local=load_vocoder_from_local,
        vocoder_name=vocoder_name,
        target_rms=target_rms,
        cross_fade_duration=cross_fade_duration,
        nfe_step=nfe_step,
        cfg_strength=cfg_strength,
        sway_sampling_coef=sway_sampling_coef,
        speed=speed,
        fix_duration=fix_duration,
        device=device
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="F5-TTS Text-to-Speech Generator")
    
    # Required arguments
    parser.add_argument("gen_text", help="Text to generate speech for")
    parser.add_argument("output_path", help="Output path for the generated audio file")
    
    # Basic optional arguments
    parser.add_argument("--ref_audio", help="Path to reference audio file")
    parser.add_argument("--ref_text", help="Reference text")
    parser.add_argument("--model", default=F5TTSWrapper.DEFAULT_MODEL, help="F5-TTS model to use")
    parser.add_argument("--temp_dir", help="Temporary directory for CLI output")
    
    # F5-TTS CLI optional arguments
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--model_cfg", help="Path to F5-TTS model config file (.yaml)")
    parser.add_argument("--ckpt_file", help="Path to model checkpoint (.pt)")
    parser.add_argument("--vocab_file", help="Path to vocab file (.txt)")
    parser.add_argument("--gen_file", help="File with text to generate (ignores gen_text if provided)")
    parser.add_argument("--output_file", help="Name of output file")
    parser.add_argument("--save_chunk", action="store_true", help="Save each audio chunk during inference")
    parser.add_argument("--no_legacy_text", action="store_true", help="Don't use lossy ASCII transliterations")
    parser.add_argument("--remove_silence", action="store_true", help="Remove long silence from output")
    parser.add_argument("--load_vocoder_from_local", action="store_true", help="Load vocoder from local directory")
    parser.add_argument("--vocoder_name", choices=["vocos", "bigvgan"], help="Vocoder to use")
    parser.add_argument("--target_rms", type=float, help="Target output speech loudness (default: 0.1)")
    parser.add_argument("--cross_fade_duration", type=float, help="Cross-fade duration in seconds (default: 0.15)")
    parser.add_argument("--nfe_step", type=int, help="Number of denoising steps (default: 32)")
    parser.add_argument("--cfg_strength", type=float, help="Classifier-free guidance strength (default: 2.0)")
    parser.add_argument("--sway_sampling_coef", type=float, help="Sway sampling coefficient (default: -1.0)")
    parser.add_argument("--speed", type=float, help="Speed of generated audio (default: 1.0)")
    parser.add_argument("--fix_duration", type=float, help="Fix total duration in seconds")
    parser.add_argument("--device", help="Device to run on")
    
    args = parser.parse_args()
    
    success = generate_speech(
        gen_text=args.gen_text,
        output_path=args.output_path,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        model=args.model,
        temp_dir=args.temp_dir,
        config=args.config,
        model_cfg=args.model_cfg,
        ckpt_file=args.ckpt_file,
        vocab_file=args.vocab_file,
        gen_file=args.gen_file,
        output_file=args.output_file,
        save_chunk=args.save_chunk,
        no_legacy_text=args.no_legacy_text,
        remove_silence=args.remove_silence,
        load_vocoder_from_local=args.load_vocoder_from_local,
        vocoder_name=args.vocoder_name,
        target_rms=args.target_rms,
        cross_fade_duration=args.cross_fade_duration,
        nfe_step=args.nfe_step,
        cfg_strength=args.cfg_strength,
        sway_sampling_coef=args.sway_sampling_coef,
        speed=args.speed,
        fix_duration=args.fix_duration,
        device=args.device
    )
    
    exit(0 if success else 1)

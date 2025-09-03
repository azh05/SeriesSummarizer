"""
Narrator package for F5-TTS text-to-speech generation.
"""

from .f5_tts_wrapper import F5TTSWrapper, generate_speech

__all__ = ['F5TTSWrapper', 'generate_speech']

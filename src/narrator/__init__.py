"""
Narrator module for text-to-speech functionality using F5-TTS.
"""

from .narrator import NarratorInterface
from . import utils

__all__ = ["NarratorInterface", "utils"]

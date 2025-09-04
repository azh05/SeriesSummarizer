"""
TV Series Summarizer - AI Agent for Processing TV Series Episode Transcripts

This package provides a comprehensive AI agent that processes TV series episode transcripts
and maintains a knowledge base about the show using ChromaDB and LangChain.
"""

from .agent import TVSeriesAgent
from .models import Episode, EpisodeInfo, Scene, Character, Relationship, PlotEvent

__version__ = "0.1.0"
__all__ = ["TVSeriesAgent", "Episode", "EpisodeInfo", "Scene", "Character", "Relationship", "PlotEvent"]

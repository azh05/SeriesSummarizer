"""Scene segmentation extractor."""

import logging
from typing import List, Dict, Any, Optional
import re

from .base_extractor import BaseExtractor
from ..models import Scene, EmotionalTone
from ..utils import load_prompt


logger = logging.getLogger(__name__)


class SceneSegmenter(BaseExtractor):
    """Segments episode transcripts into individual scenes."""
    
    def __init__(self, **kwargs):
        """Initialize scene segmenter."""
        super().__init__(**kwargs)
    
    def extract(self, transcript: str, context: Optional[Dict[str, Any]] = None) -> List[Scene]:
        """Segment transcript into scenes.
        
        Args:
            transcript: Episode transcript
            context: Context including episode_id, season, episode
            
        Returns:
            List of Scene objects
        """
        episode_id = context.get("episode_id", "unknown") if context else "unknown"
        
        # First, identify scene breaks
        scene_segments = self._identify_scene_breaks(transcript)
        
        scenes = []
        for i, segment in enumerate(scene_segments):
            scene_number = i + 1
            scene_id = f"{episode_id}_S{scene_number:03d}"
            
            # Analyze each scene segment
            scene_analysis = self._analyze_scene(segment, scene_number, episode_id)
            
            # Create Scene object
            scene = Scene(
                id=scene_id,
                episode_id=episode_id,
                scene_number=scene_number,
                content=segment,
                summary=scene_analysis.get("summary"),
                location=scene_analysis.get("location"),
                time_of_day=scene_analysis.get("time_of_day"),
                characters_present=scene_analysis.get("characters_present", []),
                key_dialogue=scene_analysis.get("key_dialogue", []),
                plot_events=scene_analysis.get("plot_events", []),
                character_developments=scene_analysis.get("character_developments", []),
                relationship_dynamics=scene_analysis.get("relationship_dynamics", []),
                emotional_tone=self._parse_emotional_tones(scene_analysis.get("emotional_tone", [])),
                mood_description=scene_analysis.get("mood_description"),
                plot_relevance=scene_analysis.get("plot_relevance", 0.5),
                foreshadowing=scene_analysis.get("foreshadowing", []),
                callbacks=scene_analysis.get("callbacks", []),
                importance_score=scene_analysis.get("importance_score", 0.5),
                themes=scene_analysis.get("themes", [])
            )
            
            scenes.append(scene)
        
        logger.info(f"Segmented transcript into {len(scenes)} scenes")
        return scenes
    
    def _identify_scene_breaks(self, transcript: str) -> List[str]:
        """Identify scene breaks in the transcript.
        
        Args:
            transcript: Full episode transcript
            
        Returns:
            List of scene segments
        """
        system_prompt = load_prompt("scene_break_identification")
        
        user_prompt = f"""Please identify scene breaks in this transcript and split it into individual scenes:

        {transcript}
        
        Return the scenes separated by "---SCENE_BREAK---" markers."""
        
        try:
            response = self._call_llm(system_prompt, user_prompt)
            
            # Split by scene break markers
            scenes = response.split("---SCENE_BREAK---")
            scenes = [scene.strip() for scene in scenes if scene.strip()]
            
            # If no scene breaks found, treat as single scene
            if len(scenes) == 0:
                scenes = [transcript]
            
            logger.debug(f"Identified {len(scenes)} scene breaks")
            return scenes
            
        except Exception as e:
            logger.error(f"Error identifying scene breaks: {e}")
            # Fallback: return transcript as single scene
            return [transcript]
    
    def _analyze_scene(self, scene_content: str, scene_number: int, episode_id: str) -> Dict[str, Any]:
        """Analyze a single scene for various elements.
        
        Args:
            scene_content: Content of the scene
            scene_number: Scene number within episode
            episode_id: Episode ID
            
        Returns:
            Dictionary with scene analysis
        """
        system_prompt = load_prompt("scene_analysis")
        
        user_prompt = f"""Analyze this scene from episode {episode_id}, scene {scene_number}:

        {scene_content}

        Provide a comprehensive analysis in JSON format."""
        
        try:
            response = self._call_llm(system_prompt, user_prompt)
            analysis = self._parse_json_response(response)
            
            # Normalize data types to match Pydantic models
            analysis = self._normalize_data_types(analysis)
            
            # Ensure all required fields have defaults
            defaults = {
                "summary": "",
                "location": None,
                "time_of_day": None,
                "characters_present": [],
                "key_dialogue": [],
                "plot_events": [],
                "character_developments": [],
                "relationship_dynamics": [],
                "emotional_tone": [],
                "mood_description": None,
                "plot_relevance": 0.5,
                "foreshadowing": [],
                "callbacks": [],
                "importance_score": 0.5,
                "themes": []
            }
            
            for key, default_value in defaults.items():
                if key not in analysis or analysis[key] is None:
                    analysis[key] = default_value
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing scene {scene_number}: {e}")
            # Return complete minimal analysis with all required fields
            return {
                "summary": "Scene analysis failed",
                "location": None,
                "time_of_day": None,
                "characters_present": [],
                "key_dialogue": [],
                "plot_events": [],
                "character_developments": [],
                "relationship_dynamics": [],
                "emotional_tone": [],
                "mood_description": None,
                "plot_relevance": 0.5,
                "foreshadowing": [],
                "callbacks": [],
                "importance_score": 0.5,
                "themes": []
            }
    
    def _parse_emotional_tones(self, tone_strings: List[str]) -> List[EmotionalTone]:
        """Parse emotional tone strings to EmotionalTone enums.
        
        Args:
            tone_strings: List of tone strings
            
        Returns:
            List of EmotionalTone enums
        """
        tones = []
        tone_mapping = {
            "happy": EmotionalTone.HAPPY,
            "sad": EmotionalTone.SAD,
            "tense": EmotionalTone.TENSE,
            "romantic": EmotionalTone.ROMANTIC,
            "comedic": EmotionalTone.COMEDIC,
            "dramatic": EmotionalTone.DRAMATIC,
            "mysterious": EmotionalTone.MYSTERIOUS,
            "action": EmotionalTone.ACTION,
            "peaceful": EmotionalTone.PEACEFUL,
            "angry": EmotionalTone.ANGRY,
            "fearful": EmotionalTone.FEARFUL,
            "nostalgic": EmotionalTone.NOSTALGIC
        }
        
        for tone_str in tone_strings:
            tone_str_lower = tone_str.lower().strip()
            if tone_str_lower in tone_mapping:
                tones.append(tone_mapping[tone_str_lower])
        
        return tones

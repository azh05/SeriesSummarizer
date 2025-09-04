"""Scene segmentation extractor."""

import logging
from typing import List, Dict, Any, Optional
import re

from .base_extractor import BaseExtractor
from ..models import Scene, EmotionalTone


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
        system_prompt = """You are an expert at identifying scene breaks in TV show transcripts. 
        
        Your task is to analyze a transcript and identify natural scene breaks. Scene breaks typically occur when:
        - Location changes (indoor to outdoor, different rooms, different buildings)
        - Time jumps (later that day, next morning, flashbacks)
        - Character group changes (different set of characters in focus)
        - Narrative shifts (different storylines, perspective changes)
        
        Look for common indicators:
        - Stage directions like "FADE IN:", "CUT TO:", "INTERIOR:", "EXTERIOR:"
        - Time indicators like "LATER", "MEANWHILE", "THE NEXT DAY"
        - Location descriptions
        - Character entrance/exit patterns
        
        Return the transcript split into scenes, with each scene as a separate item.
        Use "---SCENE_BREAK---" as the delimiter between scenes."""
        
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
        system_prompt = """You are an expert TV script analyst. Analyze the given scene and extract key information.

        CRITICAL: You must return ONLY a valid JSON object. Do not include any explanatory text before or after the JSON.

        For each scene, identify:
        1. Location/setting (where does this take place?)
        2. Time of day (if mentioned or implied)
        3. Characters present (list all characters who speak or are mentioned as present)
        4. Key dialogue (most important/memorable lines)
        5. Plot events (what happens that advances the story?)
        6. Character developments (character growth, revelations, changes)
        7. Relationship dynamics (interactions between characters, relationship changes)
        8. Emotional tone (happy, sad, tense, romantic, comedic, dramatic, mysterious, action, peaceful, angry, fearful, nostalgic)
        9. Mood description (overall atmosphere and feeling)
        10. Plot relevance (0.0-1.0, how important is this scene to the main plot?)
        11. Foreshadowing (hints about future events)
        12. Callbacks (references to previous events)
        13. Importance score (0.0-1.0, overall scene importance)
        14. Themes (what themes are explored in this scene?)
        15. Summary (2-3 sentence summary of what happens)

        Return EXACTLY this JSON structure (replace values with your analysis):
        {
          "summary": "Brief summary here",
          "location": "Location or null",
          "time_of_day": "Time or null", 
          "characters_present": ["Character1", "Character2"],
          "key_dialogue": ["Important quote 1", "Important quote 2"],
          "plot_events": ["Event 1", "Event 2"],
          "character_developments": ["Development 1", "Development 2"],
          "relationship_dynamics": ["Dynamic 1", "Dynamic 2"],
          "emotional_tone": ["tone1", "tone2"],
          "mood_description": "Mood description or null",
          "plot_relevance": 0.7,
          "foreshadowing": ["Foreshadowing 1", "Foreshadowing 2"],
          "callbacks": ["Callback 1", "Callback 2"],
          "importance_score": 0.8,
          "themes": ["Theme 1", "Theme 2"]
        }

        Use empty arrays [] for lists with no items, null for missing values, and numbers for scores."""
        
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

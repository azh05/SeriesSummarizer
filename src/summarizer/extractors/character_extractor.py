"""Character information extractor."""

import logging
from typing import List, Dict, Any, Optional

from .base_extractor import BaseExtractor
from ..models import Character, CharacterRole


logger = logging.getLogger(__name__)


class CharacterExtractor(BaseExtractor):
    """Extracts character information from scenes."""
    
    def __init__(self, **kwargs):
        """Initialize character extractor."""
        super().__init__(**kwargs)
    
    def extract(self, content: str, context: Optional[Dict[str, Any]] = None) -> List[Character]:
        """Extract character information from content.
        
        Args:
            content: Scene or episode content
            context: Additional context (episode_id, existing_characters, etc.)
            
        Returns:
            List of Character objects
        """
        episode_id = context.get("episode_id", "unknown") if context else "unknown"
        existing_characters = context.get("existing_characters", []) if context else []
        
        # First, identify all characters mentioned
        character_mentions = self._identify_characters(content)
        
        characters = []
        for char_name in character_mentions:
            # Skip if we already have this character
            if char_name in existing_characters:
                continue
                
            # Extract detailed information about this character
            char_info = self._extract_character_details(content, char_name, episode_id)
            
            # Create Character object
            character = Character(
                name=char_name,
                aliases=char_info.get("aliases", []),
                role=self._parse_character_role(char_info.get("role", "minor")),
                description=char_info.get("description"),
                occupation=char_info.get("occupation"),
                age=char_info.get("age"),
                background=char_info.get("background"),
                personality_traits=char_info.get("personality_traits", []),
                skills_abilities=char_info.get("skills_abilities", []),
                goals_motivations=char_info.get("goals_motivations", []),
                fears_weaknesses=char_info.get("fears_weaknesses", []),
                character_arc=char_info.get("character_arc"),
                important_quotes=char_info.get("important_quotes", []),
                first_appearance=episode_id,
                importance_score=char_info.get("importance_score", 0.5)
            )
            
            # Add this episode as an appearance
            character.add_appearance(episode_id)
            
            characters.append(character)
        
        logger.info(f"Extracted {len(characters)} new characters")
        return characters
    
    def extract_character_updates(self, content: str, character_name: str, 
                                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract updates to an existing character.
        
        Args:
            content: Scene or episode content
            character_name: Name of character to analyze
            context: Additional context
            
        Returns:
            Dictionary with character updates
        """
        episode_id = context.get("episode_id", "unknown") if context else "unknown"
        
        system_prompt = f"""You are analyzing a scene for character development and changes for the character "{character_name}".

        Look for:
        1. New personality traits revealed
        2. Character growth or changes
        3. New goals or motivations
        4. Important dialogue/quotes
        5. New skills or abilities revealed
        6. Background information revealed
        7. Changes in relationships
        8. Character arc progression

        Return your analysis as a JSON object with these keys:
        - new_personality_traits: List of newly revealed traits
        - character_changes: List of character developments/changes with descriptions
        - new_quotes: List of important new quotes
        - new_goals_motivations: List of newly revealed goals/motivations
        - new_skills_abilities: List of newly revealed skills/abilities
        - new_background_info: New background information revealed
        - relationship_changes: List of relationship changes involving this character
        - character_arc_progression: How the character's arc progresses in this scene"""
        
        user_prompt = f"""Analyze this content for character development of {character_name}:

        {content}

        Focus specifically on {character_name} and what new information is revealed about them or how they change/develop."""
        
        try:
            response = self._call_llm(system_prompt, user_prompt)
            updates = self._parse_json_response(response)
            
            # Normalize data types to match Pydantic models
            updates = self._normalize_data_types(updates)
            
            # Add episode context
            updates["episode_id"] = episode_id
            
            return updates
            
        except Exception as e:
            logger.error(f"Error extracting character updates for {character_name}: {e}")
            return {"episode_id": episode_id}
    
    def _identify_characters(self, content: str) -> List[str]:
        """Identify all characters mentioned in the content.
        
        Args:
            content: Content to analyze
            
        Returns:
            List of character names
        """
        system_prompt = """You are an expert at identifying characters in TV show scripts and transcripts.

        Identify ALL characters that are mentioned in the given content. This includes:
        - Characters who speak (have dialogue)
        - Characters who are present but don't speak
        - Characters who are mentioned by other characters
        - Characters who appear in stage directions

        Return ONLY the character names, one per line, using their most common/full name.
        Do not include:
        - Generic references like "the waiter", "a man", "someone"
        - Groups like "the crowd", "everyone"
        - Unclear pronouns

        Focus on named characters only."""
        
        user_prompt = f"""Identify all characters mentioned in this content:

        {content}

        List character names, one per line."""
        
        try:
            response = self._call_llm(system_prompt, user_prompt)
            characters = self._parse_list_response(response)
            
            # Clean and deduplicate
            clean_characters = []
            for char in characters:
                char = char.strip()
                if char and char not in clean_characters:
                    clean_characters.append(char)
            
            return clean_characters
            
        except Exception as e:
            logger.error(f"Error identifying characters: {e}")
            return []
    
    def _extract_character_details(self, content: str, character_name: str, episode_id: str) -> Dict[str, Any]:
        """Extract detailed information about a specific character.
        
        Args:
            content: Content containing character information
            character_name: Name of character to analyze
            episode_id: Current episode ID
            
        Returns:
            Dictionary with character details
        """
        system_prompt = f"""You are analyzing a character named "{character_name}" from their first appearance in a TV show.

        CRITICAL: You must return ONLY a valid JSON object. Do not include any explanatory text before or after the JSON.

        Extract as much information as possible about this character from the given content:

        1. Aliases/nicknames (other names they're called)
        2. Role (protagonist, antagonist, supporting, minor, guest, recurring)
        3. Physical description (if mentioned)
        4. Occupation/job (if mentioned)
        5. Age (if mentioned or can be estimated - return as STRING)
        6. Background/history (if revealed)
        7. Personality traits (what kind of person are they?)
        8. Skills/abilities (what are they good at?)
        9. Goals/motivations (what do they want?)
        10. Fears/weaknesses (what are they afraid of or bad at?)
        11. Character arc (what journey might they be on?)
        12. Important quotes (memorable things they say)
        13. Importance score (0.0-1.0, how important do they seem to the story?)

        Return EXACTLY this JSON structure (replace values with your analysis):
        {{
          "aliases": ["Nickname1", "Nickname2"],
          "role": "supporting",
          "description": "Physical description or null",
          "occupation": "Job description or null",
          "age": "25" or null,
          "background": "Background info or null",
          "personality_traits": ["Trait1", "Trait2"],
          "skills_abilities": ["Skill1", "Skill2"],
          "goals_motivations": ["Goal1", "Goal2"],
          "fears_weaknesses": ["Fear1", "Weakness1"],
          "character_arc": "Arc description or null",
          "important_quotes": ["Quote1", "Quote2"],
          "importance_score": 0.7
        }}

        Use empty arrays [] for lists with no items, null for missing values, strings for age, and numbers for importance score."""
        
        user_prompt = f"""Analyze the character "{character_name}" from this content:

        {content}

        Extract all available information about {character_name} and return as JSON."""
        
        try:
            response = self._call_llm(system_prompt, user_prompt)
            details = self._parse_json_response(response)
            
            # Normalize data types to match Pydantic models
            details = self._normalize_data_types(details)
            
            # Ensure all fields have appropriate defaults
            defaults = {
                "aliases": [],
                "role": "minor",
                "description": None,
                "occupation": None,
                "age": None,
                "background": None,
                "personality_traits": [],
                "skills_abilities": [],
                "goals_motivations": [],
                "fears_weaknesses": [],
                "character_arc": None,
                "important_quotes": [],
                "importance_score": 0.5
            }
            
            for key, default_value in defaults.items():
                if key not in details or details[key] is None:
                    if isinstance(default_value, list):
                        details[key] = []
                    else:
                        details[key] = default_value
            
            return details
            
        except Exception as e:
            logger.error(f"Error extracting details for character {character_name}: {e}")
            # Return complete minimal character data with all required fields
            return {
                "aliases": [],
                "role": "minor",
                "description": None,
                "occupation": None,
                "age": None,
                "background": None,
                "personality_traits": [],
                "skills_abilities": [],
                "goals_motivations": [],
                "fears_weaknesses": [],
                "character_arc": None,
                "important_quotes": [],
                "importance_score": 0.5
            }
    
    def _parse_character_role(self, role_string: str) -> CharacterRole:
        """Parse character role string to CharacterRole enum.
        
        Args:
            role_string: Role as string
            
        Returns:
            CharacterRole enum
        """
        role_mapping = {
            "protagonist": CharacterRole.PROTAGONIST,
            "antagonist": CharacterRole.ANTAGONIST,
            "supporting": CharacterRole.SUPPORTING,
            "minor": CharacterRole.MINOR,
            "guest": CharacterRole.GUEST,
            "recurring": CharacterRole.RECURRING
        }
        
        role_lower = role_string.lower().strip()
        return role_mapping.get(role_lower, CharacterRole.MINOR)

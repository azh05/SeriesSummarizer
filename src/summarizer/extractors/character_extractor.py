"""Character information extractor."""

import logging
from typing import List, Dict, Any, Optional

from .base_extractor import BaseExtractor
from ..models import Character, CharacterRole
from ..utils import load_prompt, load_prompt_template


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
        
        system_prompt = load_prompt_template("character_development_analysis", character_name=character_name)
        
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
        system_prompt = load_prompt("character_identification")
        
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
        system_prompt = load_prompt_template("character_detail_extraction", character_name=character_name)
        
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

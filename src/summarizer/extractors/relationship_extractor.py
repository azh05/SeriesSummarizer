"""Relationship extractor for character interactions."""

import logging
from typing import List, Dict, Any, Optional, Tuple

from .base_extractor import BaseExtractor
from ..models import Relationship, RelationshipType, RelationshipStatus
from ..utils import load_prompt, load_prompt_template


logger = logging.getLogger(__name__)


class RelationshipExtractor(BaseExtractor):
    """Extracts relationship information between characters."""
    
    def __init__(self, **kwargs):
        """Initialize relationship extractor."""
        super().__init__(**kwargs)
    
    def extract(self, content: str, context: Optional[Dict[str, Any]] = None) -> List[Relationship]:
        """Extract relationships from content.
        
        Args:
            content: Scene or episode content
            context: Additional context (episode_id, characters_present, etc.)
            
        Returns:
            List of Relationship objects
        """
        episode_id = context.get("episode_id", "unknown") if context else "unknown"
        characters_present = context.get("characters_present", []) if context else []
        
        if len(characters_present) < 2:
            return []
        
        # Identify character pairs and their relationships
        relationship_pairs = self._identify_relationship_pairs(content, characters_present)
        
        relationships = []
        for char1, char2 in relationship_pairs:
            # Extract relationship details
            rel_details = self._extract_relationship_details(content, char1, char2, episode_id)
            
            if rel_details:
                # Create Relationship object
                relationship = Relationship(
                    id=f"{char1}_{char2}".replace(" ", "_").lower(),
                    character1=char1,
                    character2=char2,
                    relationship_type=self._parse_relationship_type(rel_details.get("type", "acquaintance")),
                    current_status=self._parse_relationship_status(rel_details.get("status", "unknown")),
                    description=rel_details.get("description"),
                    how_they_met=rel_details.get("how_they_met"),
                    relationship_dynamic=rel_details.get("dynamic"),
                    first_interaction=episode_id,
                    importance_score=rel_details.get("importance_score", 0.5),
                    emotional_intensity=rel_details.get("emotional_intensity", 0.5)
                )
                
                # Add key dialogue and scenes
                if rel_details.get("key_dialogue"):
                    for dialogue in rel_details["key_dialogue"]:
                        relationship.add_dialogue(dialogue)
                
                relationships.append(relationship)
        
        logger.info(f"Extracted {len(relationships)} relationships")
        return relationships
    
    def extract_relationship_updates(self, content: str, character1: str, character2: str,
                                   context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract updates to an existing relationship.
        
        Args:
            content: Scene or episode content
            character1: First character name
            character2: Second character name
            context: Additional context
            
        Returns:
            Dictionary with relationship updates
        """
        episode_id = context.get("episode_id", "unknown") if context else "unknown"
        scene_id = context.get("scene_id") if context else None
        
        system_prompt = load_prompt_template("relationship_updates_analysis", character1=character1, character2=character2)
        
        user_prompt = f"""Analyze the relationship between {character1} and {character2} in this content:

        {content}

        Focus on their interactions and any changes in their relationship."""
        
        try:
            response = self._call_llm(system_prompt, user_prompt)
            updates = self._parse_json_response(response)
            
            # Add context
            updates["episode_id"] = episode_id
            updates["scene_id"] = scene_id
            
            return updates
            
        except Exception as e:
            logger.error(f"Error extracting relationship updates for {character1}-{character2}: {e}")
            return {"episode_id": episode_id, "scene_id": scene_id}
    
    def _identify_relationship_pairs(self, content: str, characters: List[str]) -> List[Tuple[str, str]]:
        """Identify character pairs that interact in the content.
        
        Args:
            content: Content to analyze
            characters: List of characters present
            
        Returns:
            List of character pairs that interact
        """
        if len(characters) < 2:
            return []
        
        system_prompt = load_prompt("relationship_pairs_identification")
        
        characters_list = "\n".join(f"- {char}" for char in characters)
        user_prompt = f"""Identify which character pairs interact in this content:

        Available characters:
        {characters_list}

        Content:
        {content}

        Return interacting pairs in the format: Character1 | Character2"""
        
        try:
            response = self._call_llm(system_prompt, user_prompt)
            
            pairs = []
            for line in response.strip().split('\n'):
                if '|' in line:
                    parts = line.split('|')
                    if len(parts) == 2:
                        char1 = parts[0].strip()
                        char2 = parts[1].strip()
                        
                        # Validate characters are in our list
                        if char1 in characters and char2 in characters and char1 != char2:
                            # Sort to ensure consistent ordering
                            pair = tuple(sorted([char1, char2]))
                            if pair not in pairs:
                                pairs.append(pair)
            
            return pairs
            
        except Exception as e:
            logger.error(f"Error identifying relationship pairs: {e}")
            return []
    
    def _extract_relationship_details(self, content: str, char1: str, char2: str, episode_id: str) -> Optional[Dict[str, Any]]:
        """Extract detailed relationship information between two characters.
        
        Args:
            content: Content containing the interaction
            char1: First character name
            char2: Second character name
            episode_id: Current episode ID
            
        Returns:
            Dictionary with relationship details or None
        """
        system_prompt = load_prompt_template("relationship_detail_extraction", char1=char1, char2=char2)
        
        user_prompt = f"""Analyze the relationship between {char1} and {char2} from this content:

        {content}

        Focus on their interactions and determine their relationship type and dynamic."""
        
        try:
            response = self._call_llm(system_prompt, user_prompt)
            
            # Handle null response
            if response.strip().lower() in ['null', 'none', '{}']:
                return None
            
            details = self._parse_json_response(response)
            
            # Ensure required fields
            if not details.get("type"):
                details["type"] = "acquaintance"
            if not details.get("status"):
                details["status"] = "unknown"
            if "importance_score" not in details:
                details["importance_score"] = 0.5
            if "emotional_intensity" not in details:
                details["emotional_intensity"] = 0.5
            
            return details
            
        except Exception as e:
            logger.error(f"Error extracting relationship details for {char1}-{char2}: {e}")
            return None
    
    def _parse_relationship_type(self, type_string: str) -> RelationshipType:
        """Parse relationship type string to RelationshipType enum.
        
        Args:
            type_string: Type as string
            
        Returns:
            RelationshipType enum
        """
        type_mapping = {
            "family": RelationshipType.FAMILY,
            "romantic": RelationshipType.ROMANTIC,
            "friendship": RelationshipType.FRIENDSHIP,
            "rivalry": RelationshipType.RIVALRY,
            "professional": RelationshipType.PROFESSIONAL,
            "mentor_student": RelationshipType.MENTOR_STUDENT,
            "enemy": RelationshipType.ENEMY,
            "acquaintance": RelationshipType.ACQUAINTANCE,
            "alliance": RelationshipType.ALLIANCE,
            "complicated": RelationshipType.COMPLICATED
        }
        
        type_lower = type_string.lower().strip()
        return type_mapping.get(type_lower, RelationshipType.ACQUAINTANCE)
    
    def _parse_relationship_status(self, status_string: str) -> RelationshipStatus:
        """Parse relationship status string to RelationshipStatus enum.
        
        Args:
            status_string: Status as string
            
        Returns:
            RelationshipStatus enum
        """
        status_mapping = {
            "developing": RelationshipStatus.DEVELOPING,
            "established": RelationshipStatus.ESTABLISHED,
            "strained": RelationshipStatus.STRAINED,
            "broken": RelationshipStatus.BROKEN,
            "reconciled": RelationshipStatus.RECONCILED,
            "ended": RelationshipStatus.ENDED,
            "unknown": RelationshipStatus.UNKNOWN
        }
        
        status_lower = status_string.lower().strip()
        return status_mapping.get(status_lower, RelationshipStatus.UNKNOWN)

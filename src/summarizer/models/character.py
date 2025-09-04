"""Character data model."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class CharacterRole(str, Enum):
    """Character role in the series."""
    PROTAGONIST = "protagonist"
    ANTAGONIST = "antagonist"
    SUPPORTING = "supporting"
    MINOR = "minor"
    GUEST = "guest"
    RECURRING = "recurring"


class Character(BaseModel):
    """Character data model."""
    name: str = Field(..., description="Character's full name")
    aliases: List[str] = Field(default_factory=list, description="Alternative names/nicknames")
    role: CharacterRole = Field(default=CharacterRole.MINOR, description="Character's role in series")
    
    # Basic info
    description: Optional[str] = Field(None, description="Physical description")
    occupation: Optional[str] = Field(None, description="Character's job/occupation")
    age: Optional[str] = Field(None, description="Character's age (can be approximate)")
    background: Optional[str] = Field(None, description="Character background/history")
    
    # Character traits
    personality_traits: List[str] = Field(default_factory=list, description="Key personality traits")
    skills_abilities: List[str] = Field(default_factory=list, description="Special skills or abilities")
    goals_motivations: List[str] = Field(default_factory=list, description="Character goals and motivations")
    fears_weaknesses: List[str] = Field(default_factory=list, description="Character fears and weaknesses")
    
    # Story elements
    character_arc: Optional[str] = Field(None, description="Overall character development arc")
    important_quotes: List[str] = Field(default_factory=list, description="Memorable quotes")
    key_scenes: List[str] = Field(default_factory=list, description="Scene IDs of important character moments")
    
    # Appearance tracking
    first_appearance: Optional[str] = Field(None, description="Episode ID of first appearance")
    last_appearance: Optional[str] = Field(None, description="Episode ID of last appearance")
    episode_appearances: List[str] = Field(default_factory=list, description="All episode IDs character appears in")
    
    # Development tracking
    character_changes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Timeline of character changes with episode references"
    )
    relationships: List[str] = Field(default_factory=list, description="IDs of relationships this character is in")
    
    # Analysis
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Character importance (0-1)")
    screen_time_estimate: Optional[float] = Field(None, description="Estimated screen time percentage")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now, description="When character was first added")
    updated_at: datetime = Field(default_factory=datetime.now, description="When character was last updated")

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def add_alias(self, alias: str) -> None:
        """Add an alias for this character."""
        if alias not in self.aliases and alias != self.name:
            self.aliases.append(alias)

    def add_personality_trait(self, trait: str) -> None:
        """Add a personality trait."""
        if trait not in self.personality_traits:
            self.personality_traits.append(trait)

    def add_quote(self, quote: str) -> None:
        """Add an important quote."""
        if quote not in self.important_quotes:
            self.important_quotes.append(quote)

    def add_appearance(self, episode_id: str) -> None:
        """Add an episode appearance."""
        if episode_id not in self.episode_appearances:
            self.episode_appearances.append(episode_id)
        
        # Update first/last appearance
        if not self.first_appearance:
            self.first_appearance = episode_id
        self.last_appearance = episode_id

    def add_character_change(self, change_description: str, episode_id: str, scene_id: Optional[str] = None) -> None:
        """Add a character development/change."""
        change_entry = {
            "description": change_description,
            "episode_id": episode_id,
            "scene_id": scene_id,
            "timestamp": datetime.now().isoformat()
        }
        self.character_changes.append(change_entry)
        self.updated_at = datetime.now()

    def add_relationship(self, relationship_id: str) -> None:
        """Add a relationship ID."""
        if relationship_id not in self.relationships:
            self.relationships.append(relationship_id)

    def get_character_journey(self) -> List[Dict[str, Any]]:
        """Get chronological character development journey."""
        return sorted(self.character_changes, key=lambda x: x["episode_id"])

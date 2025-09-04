"""Relationship data model."""

from datetime import datetime
from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field


class RelationshipType(str, Enum):
    """Types of relationships between characters."""
    FAMILY = "family"
    ROMANTIC = "romantic"
    FRIENDSHIP = "friendship"
    RIVALRY = "rivalry"
    PROFESSIONAL = "professional"
    MENTOR_STUDENT = "mentor_student"
    ENEMY = "enemy"
    ACQUAINTANCE = "acquaintance"
    ALLIANCE = "alliance"
    COMPLICATED = "complicated"


class RelationshipStatus(str, Enum):
    """Current status of the relationship."""
    DEVELOPING = "developing"
    ESTABLISHED = "established"
    STRAINED = "strained"
    BROKEN = "broken"
    RECONCILED = "reconciled"
    ENDED = "ended"
    UNKNOWN = "unknown"


class RelationshipChange(BaseModel):
    """A change in the relationship over time."""
    episode_id: str = Field(..., description="Episode where change occurred")
    scene_id: Optional[str] = Field(None, description="Specific scene of change")
    old_status: Optional[RelationshipStatus] = Field(None, description="Previous status")
    new_status: RelationshipStatus = Field(..., description="New status")
    description: str = Field(..., description="Description of what changed")
    key_moment: Optional[str] = Field(None, description="Key dialogue or action that caused change")
    timestamp: datetime = Field(default_factory=datetime.now, description="When change was recorded")


class Relationship(BaseModel):
    """Relationship between two characters."""
    id: str = Field(..., description="Unique relationship identifier")
    character1: str = Field(..., description="First character name")
    character2: str = Field(..., description="Second character name")
    relationship_type: RelationshipType = Field(..., description="Type of relationship")
    current_status: RelationshipStatus = Field(default=RelationshipStatus.UNKNOWN, description="Current status")
    
    # Relationship details
    description: Optional[str] = Field(None, description="Description of the relationship")
    how_they_met: Optional[str] = Field(None, description="How the characters first met")
    relationship_dynamic: Optional[str] = Field(None, description="Overall dynamic between characters")
    
    # Key moments
    first_interaction: Optional[str] = Field(None, description="Episode ID of first interaction")
    key_scenes: List[str] = Field(default_factory=list, description="Important scenes for this relationship")
    important_dialogue: List[str] = Field(default_factory=list, description="Key dialogue between characters")
    
    # Development tracking
    relationship_changes: List[RelationshipChange] = Field(
        default_factory=list,
        description="Timeline of relationship changes"
    )
    conflict_patterns: List[str] = Field(default_factory=list, description="Recurring conflict themes")
    resolution_patterns: List[str] = Field(default_factory=list, description="How conflicts are typically resolved")
    
    # Analysis
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Relationship importance (0-1)")
    screen_time_together: Optional[float] = Field(None, description="Estimated time characters spend together")
    emotional_intensity: float = Field(default=0.5, ge=0.0, le=1.0, description="Emotional intensity of relationship")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now, description="When relationship was first recorded")
    updated_at: datetime = Field(default_factory=datetime.now, description="When relationship was last updated")

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    @property
    def relationship_id(self) -> str:
        """Generate relationship ID from character names."""
        # Sort names to ensure consistent ID regardless of order
        names = sorted([self.character1, self.character2])
        return f"{names[0]}_{names[1]}".replace(" ", "_").lower()

    def add_key_scene(self, scene_id: str) -> None:
        """Add a key scene for this relationship."""
        if scene_id not in self.key_scenes:
            self.key_scenes.append(scene_id)

    def add_dialogue(self, dialogue: str) -> None:
        """Add important dialogue between the characters."""
        if dialogue not in self.important_dialogue:
            self.important_dialogue.append(dialogue)

    def add_relationship_change(
        self,
        episode_id: str,
        new_status: RelationshipStatus,
        description: str,
        scene_id: Optional[str] = None,
        key_moment: Optional[str] = None
    ) -> None:
        """Add a change in the relationship."""
        change = RelationshipChange(
            episode_id=episode_id,
            scene_id=scene_id,
            old_status=self.current_status,
            new_status=new_status,
            description=description,
            key_moment=key_moment
        )
        self.relationship_changes.append(change)
        self.current_status = new_status
        self.updated_at = datetime.now()

    def get_relationship_timeline(self) -> List[RelationshipChange]:
        """Get chronological timeline of relationship changes."""
        return sorted(self.relationship_changes, key=lambda x: x.episode_id)

    def involves_character(self, character_name: str) -> bool:
        """Check if this relationship involves a specific character."""
        return character_name in [self.character1, self.character2]

    def get_other_character(self, character_name: str) -> Optional[str]:
        """Get the other character in this relationship."""
        if character_name == self.character1:
            return self.character2
        elif character_name == self.character2:
            return self.character1
        return None

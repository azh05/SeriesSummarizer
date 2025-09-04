"""Scene data model."""

from datetime import datetime
from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field


class EmotionalTone(str, Enum):
    """Emotional tone of a scene."""
    HAPPY = "happy"
    SAD = "sad"
    TENSE = "tense"
    ROMANTIC = "romantic"
    COMEDIC = "comedic"
    DRAMATIC = "dramatic"
    MYSTERIOUS = "mysterious"
    ACTION = "action"
    PEACEFUL = "peaceful"
    ANGRY = "angry"
    FEARFUL = "fearful"
    NOSTALGIC = "nostalgic"


class Scene(BaseModel):
    """Individual scene data model."""
    id: str = Field(..., description="Unique scene identifier (e.g., 'S01E01_S001')")
    episode_id: str = Field(..., description="Parent episode ID")
    scene_number: int = Field(..., ge=1, description="Scene number within episode")
    
    # Content
    content: str = Field(..., description="Scene transcript/content")
    summary: Optional[str] = Field(None, description="Scene summary")
    
    # Context
    location: Optional[str] = Field(None, description="Scene location/setting")
    time_of_day: Optional[str] = Field(None, description="Time when scene occurs")
    characters_present: List[str] = Field(default_factory=list, description="Characters in this scene")
    
    # Analysis
    key_dialogue: List[str] = Field(default_factory=list, description="Important dialogue lines")
    plot_events: List[str] = Field(default_factory=list, description="Plot events that occur")
    character_developments: List[str] = Field(default_factory=list, description="Character development moments")
    relationship_dynamics: List[str] = Field(default_factory=list, description="Relationship interactions")
    
    # Tone and mood
    emotional_tone: List[EmotionalTone] = Field(default_factory=list, description="Emotional tones present")
    mood_description: Optional[str] = Field(None, description="Detailed mood description")
    
    # Story structure
    plot_relevance: float = Field(default=0.5, ge=0.0, le=1.0, description="Relevance to main plot (0-1)")
    foreshadowing: List[str] = Field(default_factory=list, description="Foreshadowing elements")
    callbacks: List[str] = Field(default_factory=list, description="References to previous events")
    
    # Metadata
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Scene importance (0-1)")
    themes: List[str] = Field(default_factory=list, description="Themes explored in scene")
    processed_at: datetime = Field(default_factory=datetime.now, description="When scene was processed")

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    @property
    def scene_id(self) -> str:
        """Generate scene ID from episode and scene number."""
        return f"{self.episode_id}_S{self.scene_number:03d}"

    def add_character(self, character_name: str) -> None:
        """Add a character present in this scene."""
        if character_name not in self.characters_present:
            self.characters_present.append(character_name)

    def add_key_dialogue(self, dialogue: str) -> None:
        """Add important dialogue from this scene."""
        if dialogue not in self.key_dialogue:
            self.key_dialogue.append(dialogue)

    def add_plot_event(self, event: str) -> None:
        """Add a plot event that occurs in this scene."""
        if event not in self.plot_events:
            self.plot_events.append(event)

    def add_emotional_tone(self, tone: EmotionalTone) -> None:
        """Add an emotional tone to this scene."""
        if tone not in self.emotional_tone:
            self.emotional_tone.append(tone)

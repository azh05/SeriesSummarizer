"""Plot event data model."""

from datetime import datetime
from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Types of plot events."""
    MAIN_PLOT = "main_plot"
    SUBPLOT = "subplot"
    CHARACTER_DEVELOPMENT = "character_development"
    WORLD_BUILDING = "world_building"
    MYSTERY_CLUE = "mystery_clue"
    MYSTERY_RESOLUTION = "mystery_resolution"
    CONFLICT_INTRODUCTION = "conflict_introduction"
    CONFLICT_ESCALATION = "conflict_escalation"
    CONFLICT_RESOLUTION = "conflict_resolution"
    REVELATION = "revelation"
    TWIST = "twist"
    CLIFFHANGER = "cliffhanger"
    FLASHBACK = "flashback"
    FORESHADOWING = "foreshadowing"
    CALLBACK = "callback"


class EventImportance(str, Enum):
    """Importance level of the event."""
    CRITICAL = "critical"  # Major plot points, series-changing events
    HIGH = "high"         # Important developments
    MEDIUM = "medium"     # Significant but not crucial
    LOW = "low"          # Minor events


class PlotEvent(BaseModel):
    """Plot event data model."""
    id: str = Field(..., description="Unique event identifier")
    title: str = Field(..., description="Brief title/name of the event")
    description: str = Field(..., description="Detailed description of what happened")
    event_type: EventType = Field(..., description="Type of plot event")
    importance: EventImportance = Field(default=EventImportance.MEDIUM, description="Event importance level")
    
    # Location in series
    episode_id: str = Field(..., description="Episode where event occurs")
    scene_id: Optional[str] = Field(None, description="Specific scene of the event")
    
    # Characters and relationships
    characters_involved: List[str] = Field(default_factory=list, description="Characters involved in the event")
    relationships_affected: List[str] = Field(default_factory=list, description="Relationship IDs affected")
    
    # Plot connections
    plot_arc: Optional[str] = Field(None, description="Which plot arc this event belongs to")
    causes: List[str] = Field(default_factory=list, description="Event IDs that led to this event")
    consequences: List[str] = Field(default_factory=list, description="Event IDs that result from this event")
    related_events: List[str] = Field(default_factory=list, description="Other related event IDs")
    
    # Story structure
    setup_episodes: List[str] = Field(default_factory=list, description="Episodes that set up this event")
    payoff_episodes: List[str] = Field(default_factory=list, description="Episodes where this event pays off")
    foreshadowing_clues: List[str] = Field(default_factory=list, description="Earlier clues that hinted at this")
    
    # Analysis
    themes: List[str] = Field(default_factory=list, description="Themes explored through this event")
    emotional_impact: float = Field(default=0.5, ge=0.0, le=1.0, description="Emotional impact on audience")
    plot_significance: float = Field(default=0.5, ge=0.0, le=1.0, description="Significance to overall plot")
    
    # Mystery/revelation specific
    mystery_elements: List[str] = Field(default_factory=list, description="Mystery elements introduced/resolved")
    reveals_information: List[str] = Field(default_factory=list, description="Information revealed by this event")
    questions_raised: List[str] = Field(default_factory=list, description="Questions raised by this event")
    questions_answered: List[str] = Field(default_factory=list, description="Questions answered by this event")
    
    # Metadata
    tags: List[str] = Field(default_factory=list, description="Additional tags for categorization")
    notes: Optional[str] = Field(None, description="Additional analysis notes")
    created_at: datetime = Field(default_factory=datetime.now, description="When event was recorded")
    updated_at: datetime = Field(default_factory=datetime.now, description="When event was last updated")

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def add_character(self, character_name: str) -> None:
        """Add a character involved in this event."""
        if character_name not in self.characters_involved:
            self.characters_involved.append(character_name)

    def add_consequence(self, event_id: str) -> None:
        """Add a consequence event."""
        if event_id not in self.consequences:
            self.consequences.append(event_id)

    def add_cause(self, event_id: str) -> None:
        """Add a causal event."""
        if event_id not in self.causes:
            self.causes.append(event_id)

    def add_related_event(self, event_id: str) -> None:
        """Add a related event."""
        if event_id not in self.related_events:
            self.related_events.append(event_id)

    def add_foreshadowing_clue(self, clue: str) -> None:
        """Add a foreshadowing clue."""
        if clue not in self.foreshadowing_clues:
            self.foreshadowing_clues.append(clue)

    def add_mystery_element(self, element: str) -> None:
        """Add a mystery element."""
        if element not in self.mystery_elements:
            self.mystery_elements.append(element)

    def add_theme(self, theme: str) -> None:
        """Add a theme explored in this event."""
        if theme not in self.themes:
            self.themes.append(theme)

    def add_tag(self, tag: str) -> None:
        """Add a tag for categorization."""
        if tag not in self.tags:
            self.tags.append(tag)

    def is_mystery_related(self) -> bool:
        """Check if this event is related to a mystery."""
        return self.event_type in [EventType.MYSTERY_CLUE, EventType.MYSTERY_RESOLUTION] or bool(self.mystery_elements)

    def is_major_event(self) -> bool:
        """Check if this is a major plot event."""
        return self.importance in [EventImportance.CRITICAL, EventImportance.HIGH] or self.plot_significance >= 0.7

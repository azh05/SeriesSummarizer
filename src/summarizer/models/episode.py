"""Episode data model."""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class EpisodeInfo(BaseModel):
    """Episode metadata information."""
    season: int = Field(..., ge=1, description="Season number")
    episode: int = Field(..., ge=1, description="Episode number within season")
    title: str = Field(..., description="Episode title")
    air_date: Optional[str] = Field(None, description="Original air date (YYYY-MM-DD)")
    duration: Optional[int] = Field(None, description="Episode duration in minutes")
    description: Optional[str] = Field(None, description="Episode description/synopsis")


class Episode(BaseModel):
    """Complete episode data model."""
    id: str = Field(..., description="Unique episode identifier (e.g., 'S01E01')")
    info: EpisodeInfo = Field(..., description="Episode metadata")
    transcript: str = Field(..., description="Full episode transcript")
    scenes: List[str] = Field(default_factory=list, description="List of scene IDs")
    characters_introduced: List[str] = Field(default_factory=list, description="New characters in this episode")
    plot_arcs: List[str] = Field(default_factory=list, description="Plot arcs active in this episode")
    themes: List[str] = Field(default_factory=list, description="Major themes explored")
    summary: Optional[str] = Field(None, description="Generated episode summary")
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Overall episode importance (0-1)")
    processed_at: datetime = Field(default_factory=datetime.now, description="When episode was processed")

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    @property
    def episode_id(self) -> str:
        """Generate episode ID from season and episode numbers."""
        return f"S{self.info.season:02d}E{self.info.episode:02d}"

    def add_scene(self, scene_id: str) -> None:
        """Add a scene ID to this episode."""
        if scene_id not in self.scenes:
            self.scenes.append(scene_id)

    def add_character(self, character_name: str) -> None:
        """Add a character introduced in this episode."""
        if character_name not in self.characters_introduced:
            self.characters_introduced.append(character_name)

    def add_plot_arc(self, arc_name: str) -> None:
        """Add a plot arc active in this episode."""
        if arc_name not in self.plot_arcs:
            self.plot_arcs.append(arc_name)

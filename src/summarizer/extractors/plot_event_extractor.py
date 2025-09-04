"""Plot event extractor for story elements."""

import logging
from typing import List, Dict, Any, Optional

from .base_extractor import BaseExtractor
from ..models import PlotEvent, EventType, EventImportance


logger = logging.getLogger(__name__)


class PlotEventExtractor(BaseExtractor):
    """Extracts plot events and story elements."""
    
    def __init__(self, **kwargs):
        """Initialize plot event extractor."""
        super().__init__(**kwargs)
    
    def extract(self, content: str, context: Optional[Dict[str, Any]] = None) -> List[PlotEvent]:
        """Extract plot events from content.
        
        Args:
            content: Scene or episode content
            context: Additional context (episode_id, scene_id, characters_present, etc.)
            
        Returns:
            List of PlotEvent objects
        """
        episode_id = context.get("episode_id", "unknown") if context else "unknown"
        scene_id = context.get("scene_id") if context else None
        characters_present = context.get("characters_present", []) if context else []
        
        # Identify plot events in the content
        events_data = self._identify_plot_events(content, episode_id, scene_id)
        
        plot_events = []
        for i, event_data in enumerate(events_data):
            # Generate event ID
            event_id = f"{episode_id}_E{i+1:03d}"
            if scene_id:
                event_id = f"{scene_id}_E{i+1:03d}"
            
            # Create PlotEvent object
            plot_event = PlotEvent(
                id=event_id,
                title=event_data.get("title", f"Event {i+1}"),
                description=event_data.get("description", ""),
                event_type=self._parse_event_type(event_data.get("type", "main_plot")),
                importance=self._parse_event_importance(event_data.get("importance", "medium")),
                episode_id=episode_id,
                scene_id=scene_id,
                characters_involved=event_data.get("characters_involved", characters_present),
                relationships_affected=event_data.get("relationships_affected", []),
                plot_arc=event_data.get("plot_arc"),
                themes=event_data.get("themes", []),
                emotional_impact=event_data.get("emotional_impact", 0.5),
                plot_significance=event_data.get("plot_significance", 0.5),
                mystery_elements=event_data.get("mystery_elements", []),
                reveals_information=event_data.get("reveals_information", []),
                questions_raised=event_data.get("questions_raised", []),
                questions_answered=event_data.get("questions_answered", []),
                tags=event_data.get("tags", [])
            )
            
            # Add foreshadowing and setup information
            if event_data.get("foreshadowing_clues"):
                for clue in event_data["foreshadowing_clues"]:
                    plot_event.add_foreshadowing_clue(clue)
            
            plot_events.append(plot_event)
        
        logger.info(f"Extracted {len(plot_events)} plot events")
        return plot_events
    
    def extract_plot_connections(self, content: str, existing_events: List[str],
                               context: Optional[Dict[str, Any]] = None) -> Dict[str, List[str]]:
        """Extract connections between plot events.
        
        Args:
            content: Content to analyze
            existing_events: List of existing event IDs
            context: Additional context
            
        Returns:
            Dictionary with event connections (causes, consequences, related)
        """
        system_prompt = f"""You are analyzing plot connections and causality in a TV show.

        Given the current content and a list of existing plot events, identify:
        1. Causal relationships (which events led to current events)
        2. Consequences (what events result from current events)
        3. Related events (events that are connected but not directly causal)
        4. Setup/payoff relationships

        Return as JSON with these keys:
        - causes: List of event IDs that led to events in current content
        - consequences: List of event IDs that result from current content
        - related: List of event IDs that are related to current content
        - setup_for: List of event IDs that current content sets up
        - payoff_for: List of event IDs that current content pays off"""
        
        existing_events_str = "\n".join(f"- {event_id}" for event_id in existing_events)
        user_prompt = f"""Analyze plot connections for this content:

        {content}

        Existing events:
        {existing_events_str}

        Identify connections between current content and existing events."""
        
        try:
            response = self._call_llm(system_prompt, user_prompt)
            connections = self._parse_json_response(response)
            
            # Normalize data types to match expected structure
            connections = self._normalize_data_types(connections)
            
            # Ensure all fields exist
            for key in ["causes", "consequences", "related", "setup_for", "payoff_for"]:
                if key not in connections:
                    connections[key] = []
            
            return connections
            
        except Exception as e:
            logger.error(f"Error extracting plot connections: {e}")
            return {"causes": [], "consequences": [], "related": [], "setup_for": [], "payoff_for": []}
    
    def _identify_plot_events(self, content: str, episode_id: str, scene_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Identify plot events in the content.
        
        Args:
            content: Content to analyze
            episode_id: Episode ID
            scene_id: Scene ID if available
            
        Returns:
            List of plot event data dictionaries
        """
        system_prompt = """You are an expert story analyst identifying plot events in TV show content.

        Identify ALL significant plot events that occur in the given content. For each event, determine:

        1. Title (brief, descriptive name)
        2. Description (what happens)
        3. Type (main_plot, subplot, character_development, world_building, mystery_clue, mystery_resolution, 
                conflict_introduction, conflict_escalation, conflict_resolution, revelation, twist, 
                cliffhanger, flashback, foreshadowing, callback)
        4. Importance (critical, high, medium, low)
        5. Characters involved
        6. Plot arc (if part of a larger storyline)
        7. Themes explored
        8. Emotional impact (0.0-1.0)
        9. Plot significance (0.0-1.0)
        10. Mystery elements (if any)
        11. Information revealed
        12. Questions raised
        13. Questions answered
        14. Foreshadowing clues
        15. Tags for categorization

        Return as a JSON array of events. If no significant events occur, return an empty array."""
        
        location_info = f" in scene {scene_id}" if scene_id else ""
        user_prompt = f"""Identify plot events in this content from episode {episode_id}{location_info}:

        {content}

        Return all significant plot events as a JSON array."""
        
        try:
            response = self._call_llm(system_prompt, user_prompt)
            
            # Handle empty response
            if response.strip() in ['[]', '', 'null']:
                return []
            
            events = self._parse_json_response(response)
            
            # Normalize data types to match expected structure
            events = self._normalize_data_types(events)
            
            # Ensure it's a list
            if isinstance(events, dict):
                events = [events]
            elif not isinstance(events, list):
                return []
            
            # Validate and clean events
            clean_events = []
            for event in events:
                if isinstance(event, dict) and event.get("title") and event.get("description"):
                    # Set defaults for missing fields
                    defaults = {
                        "type": "main_plot",
                        "importance": "medium",
                        "characters_involved": [],
                        "themes": [],
                        "emotional_impact": 0.5,
                        "plot_significance": 0.5,
                        "mystery_elements": [],
                        "reveals_information": [],
                        "questions_raised": [],
                        "questions_answered": [],
                        "foreshadowing_clues": [],
                        "tags": []
                    }
                    
                    for key, default_value in defaults.items():
                        if key not in event:
                            event[key] = default_value
                    
                    clean_events.append(event)
            
            return clean_events
            
        except Exception as e:
            logger.error(f"Error identifying plot events: {e}")
            return []
    
    def _parse_event_type(self, type_string: str) -> EventType:
        """Parse event type string to EventType enum.
        
        Args:
            type_string: Type as string
            
        Returns:
            EventType enum
        """
        type_mapping = {
            "main_plot": EventType.MAIN_PLOT,
            "subplot": EventType.SUBPLOT,
            "character_development": EventType.CHARACTER_DEVELOPMENT,
            "world_building": EventType.WORLD_BUILDING,
            "mystery_clue": EventType.MYSTERY_CLUE,
            "mystery_resolution": EventType.MYSTERY_RESOLUTION,
            "conflict_introduction": EventType.CONFLICT_INTRODUCTION,
            "conflict_escalation": EventType.CONFLICT_ESCALATION,
            "conflict_resolution": EventType.CONFLICT_RESOLUTION,
            "revelation": EventType.REVELATION,
            "twist": EventType.TWIST,
            "cliffhanger": EventType.CLIFFHANGER,
            "flashback": EventType.FLASHBACK,
            "foreshadowing": EventType.FORESHADOWING,
            "callback": EventType.CALLBACK
        }
        
        type_lower = type_string.lower().strip()
        return type_mapping.get(type_lower, EventType.MAIN_PLOT)
    
    def _parse_event_importance(self, importance_string: str) -> EventImportance:
        """Parse event importance string to EventImportance enum.
        
        Args:
            importance_string: Importance as string
            
        Returns:
            EventImportance enum
        """
        importance_mapping = {
            "critical": EventImportance.CRITICAL,
            "high": EventImportance.HIGH,
            "medium": EventImportance.MEDIUM,
            "low": EventImportance.LOW
        }
        
        importance_lower = importance_string.lower().strip()
        return importance_mapping.get(importance_lower, EventImportance.MEDIUM)

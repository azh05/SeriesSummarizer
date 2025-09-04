"""Summary generation with cross-referencing capabilities."""

import logging
from typing import Dict, Any, List, Optional

from ..database import ChromaDBManager
from ..extractors.base_extractor import BaseExtractor


logger = logging.getLogger(__name__)


class SummaryGenerator(BaseExtractor):
    """Generates comprehensive summaries with cross-referencing."""
    
    def __init__(self, db_manager: ChromaDBManager, **kwargs):
        """Initialize summary generator.
        
        Args:
            db_manager: ChromaDB manager for cross-referencing
        """
        super().__init__(**kwargs)
        self.db_manager = db_manager
    
    def generate_episode_summary(self, episode_id: str) -> str:
        """Generate comprehensive episode summary with cross-references.
        
        Args:
            episode_id: Episode ID (e.g., 'S01E01')
            
        Returns:
            Comprehensive episode summary
        """
        # Get episode data
        episode_data = self.db_manager.get_episode_by_id(episode_id)
        if not episode_data:
            return f"Episode {episode_id} not found."
        
        # Get all scenes for the episode
        scenes = self.db_manager.get_scenes_for_episode(episode_id)
        
        # Get plot events for the episode
        plot_events_result = self.db_manager.query_plot_events(
            f"episode {episode_id}",
            where_filter={"episode_id": episode_id}
        )
        
        # Generate context-aware summary
        summary_context = {
            "episode_data": episode_data,
            "scenes": scenes,
            "plot_events": plot_events_result,
            "episode_id": episode_id
        }
        
        return self._generate_contextual_summary(summary_context)
    
    def generate_scene_summary(self, scene_id: str, include_context: bool = True) -> str:
        """Generate summary for a specific scene.
        
        Args:
            scene_id: Scene ID
            include_context: Whether to include cross-references and context
            
        Returns:
            Scene summary
        """
        # Get scene data from database
        try:
            scene_result = self.db_manager.collections["scenes"].get(ids=[scene_id])
            if not scene_result["documents"]:
                return f"Scene {scene_id} not found."
            
            scene_doc = scene_result["documents"][0]
            scene_metadata = scene_result["metadatas"][0]
            
        except Exception as e:
            logger.error(f"Error retrieving scene {scene_id}: {e}")
            return f"Error retrieving scene {scene_id}."
        
        if not include_context:
            return scene_metadata.get("summary", "No summary available.")
        
        # Add context and cross-references
        episode_id = scene_metadata.get("episode_id")
        characters = scene_metadata.get("characters_present", [])
        
        context_info = []
        
        # Add character context
        if characters:
            context_info.append(f"Characters: {', '.join(characters)}")
        
        # Add episode context
        if episode_id:
            context_info.append(f"Episode: {episode_id}")
        
        # Add location if available
        location = scene_metadata.get("location")
        if location:
            context_info.append(f"Location: {location}")
        
        # Combine summary with context
        base_summary = scene_metadata.get("summary", "No summary available.")
        context_str = " | ".join(context_info)
        
        return f"{base_summary}\n\nContext: {context_str}"
    
    def generate_character_profile(self, character_name: str) -> str:
        """Generate comprehensive character profile.
        
        Args:
            character_name: Name of the character
            
        Returns:
            Character profile summary
        """
        # Get character data
        character_data = self.db_manager.get_character_by_name(character_name)
        if not character_data:
            return f"Character '{character_name}' not found."
        
        character_doc = character_data["document"]
        character_metadata = character_data["metadata"]
        
        # Get character's relationships
        relationships = self.db_manager.query_relationships(
            f"relationship with {character_name}",
            where_filter={"$or": [
                {"character1": character_name},
                {"character2": character_name}
            ]}
        )
        
        # Get character's key scenes
        # Use semantic search instead of metadata filtering since ChromaDB doesn't support $contains
        scenes = self.db_manager.query_scenes(
            f"scenes with {character_name}"
        )
        
        # Generate comprehensive profile
        profile_parts = []
        
        # Basic info
        profile_parts.append(f"# Character Profile: {character_name}")
        profile_parts.append(f"Role: {character_metadata.get('role', 'Unknown')}")
        
        if character_metadata.get('first_appearance'):
            profile_parts.append(f"First Appearance: {character_metadata['first_appearance']}")
        
        # Personality and traits (extracted from document content)
        if "Personality Traits:" in character_doc:
            traits_line = [line for line in character_doc.split('\n') if 'Personality Traits:' in line]
            if traits_line:
                profile_parts.append(traits_line[0])
        
        # Relationships
        if relationships.get("documents"):
            profile_parts.append("\n## Relationships:")
            for i, rel_doc in enumerate(relationships["documents"][:5]):  # Top 5 relationships
                rel_meta = relationships["metadatas"][i]
                other_char = rel_meta.get("character1") if rel_meta.get("character2") == character_name else rel_meta.get("character2")
                rel_type = rel_meta.get("relationship_type", "unknown")
                profile_parts.append(f"- {other_char}: {rel_type}")
        
        # Key appearances (use scenes data instead of metadata)
        if scenes.get("documents"):
            unique_episodes = set()
            for i, scene_doc in enumerate(scenes["documents"]):
                scene_meta = scenes["metadatas"][i]
                if scene_meta.get("episode_id"):
                    unique_episodes.add(scene_meta["episode_id"])
            
            if unique_episodes:
                appearances = sorted(list(unique_episodes))
                profile_parts.append(f"\nAppearances: {len(appearances)} episodes")
                if len(appearances) <= 10:
                    profile_parts.append(f"Episodes: {', '.join(appearances)}")
                else:
                    profile_parts.append(f"Episodes: {', '.join(appearances[:10])}... and {len(appearances) - 10} more")
        
        return "\n".join(profile_parts)
    
    def generate_relationship_summary(self, character1: str, character2: str) -> str:
        """Generate relationship summary between two characters.
        
        Args:
            character1: First character name
            character2: Second character name
            
        Returns:
            Relationship summary
        """
        # Generate relationship ID
        rel_id = f"{character1}_{character2}".replace(" ", "_").lower()
        alt_rel_id = f"{character2}_{character1}".replace(" ", "_").lower()
        
        # Try to get relationship
        try:
            rel_result = self.db_manager.collections["relationships"].get(ids=[rel_id])
            if not rel_result["documents"]:
                rel_result = self.db_manager.collections["relationships"].get(ids=[alt_rel_id])
            
            if not rel_result["documents"]:
                return f"No relationship found between {character1} and {character2}."
            
            rel_doc = rel_result["documents"][0]
            rel_metadata = rel_result["metadatas"][0]
            
        except Exception as e:
            logger.error(f"Error retrieving relationship {character1}-{character2}: {e}")
            return f"Error retrieving relationship between {character1} and {character2}."
        
        # Format relationship summary
        summary_parts = []
        summary_parts.append(f"# Relationship: {character1} & {character2}")
        summary_parts.append(f"Type: {rel_metadata.get('relationship_type', 'Unknown')}")
        summary_parts.append(f"Status: {rel_metadata.get('current_status', 'Unknown')}")
        
        if rel_metadata.get('first_interaction'):
            summary_parts.append(f"First Interaction: {rel_metadata['first_interaction']}")
        
        if rel_metadata.get('key_scenes'):
            key_scenes = rel_metadata['key_scenes']
            summary_parts.append(f"Key Scenes: {', '.join(key_scenes[:5])}")
        
        return "\n".join(summary_parts)
    
    def generate_plot_arc_summary(self, arc_name: str) -> str:
        """Generate summary of a specific plot arc.
        
        Args:
            arc_name: Name of the plot arc
            
        Returns:
            Plot arc summary
        """
        # Query for events in this plot arc
        arc_events = self.db_manager.query_plot_events(
            f"plot arc {arc_name}",
            where_filter={"plot_arc": arc_name}
        )
        
        if not arc_events.get("documents"):
            return f"No events found for plot arc '{arc_name}'."
        
        # Sort events by episode
        events_data = []
        for i, doc in enumerate(arc_events["documents"]):
            metadata = arc_events["metadatas"][i]
            events_data.append({
                "document": doc,
                "metadata": metadata,
                "episode_id": metadata.get("episode_id", "")
            })
        
        # Sort by episode ID
        events_data.sort(key=lambda x: x["episode_id"])
        
        # Generate arc summary
        summary_parts = []
        summary_parts.append(f"# Plot Arc: {arc_name}")
        summary_parts.append(f"Total Events: {len(events_data)}")
        
        if events_data:
            first_episode = events_data[0]["episode_id"]
            last_episode = events_data[-1]["episode_id"]
            summary_parts.append(f"Span: {first_episode} to {last_episode}")
        
        # Add key events
        summary_parts.append("\n## Key Events:")
        for event_data in events_data[:10]:  # Top 10 events
            metadata = event_data["metadata"]
            title = metadata.get("title", "Untitled Event")
            episode = metadata.get("episode_id", "Unknown")
            importance = metadata.get("importance", "medium")
            summary_parts.append(f"- {episode}: {title} ({importance})")
        
        return "\n".join(summary_parts)
    
    def _generate_contextual_summary(self, context: Dict[str, Any]) -> str:
        """Generate contextual summary using LLM.
        
        Args:
            context: Context dictionary with episode data
            
        Returns:
            Generated summary
        """
        episode_data = context["episode_data"]
        scenes = context["scenes"]
        episode_id = context["episode_id"]
        
        # Prepare context information
        episode_metadata = episode_data["metadata"]
        title = episode_metadata.get("title", "Unknown Title")
        season = episode_metadata.get("season", "?")
        episode_num = episode_metadata.get("episode", "?")
        
        # Collect scene summaries
        scene_summaries = []
        for scene_data in scenes[:10]:  # Limit to first 10 scenes
            scene_meta = scene_data["metadata"]
            scene_summary = scene_meta.get("summary", "No summary")
            scene_num = scene_meta.get("scene_number", "?")
            scene_summaries.append(f"Scene {scene_num}: {scene_summary}")
        
        system_prompt = """You are an expert TV show analyst creating comprehensive episode summaries.

        Create a detailed, engaging summary that includes:
        1. What happens in the episode (main plot points)
        2. Character developments and interactions
        3. Important dialogue or moments
        4. How this episode advances the overall story
        5. Key themes explored
        6. Connections to previous episodes (if apparent)
        7. Setup for future episodes (if apparent)

        Make the summary informative but engaging, as if writing for fans of the show."""
        
        scenes_text = "\n".join(scene_summaries) if scene_summaries else "No scene summaries available."
        
        user_prompt = f"""Create a comprehensive summary for:

        Episode: Season {season}, Episode {episode_num} - "{title}"
        Episode ID: {episode_id}

        Scene Summaries:
        {scenes_text}

        Generate an engaging, detailed summary that captures the essence of this episode."""
        
        try:
            summary = self._call_llm(system_prompt, user_prompt)
            return summary
        except Exception as e:
            logger.error(f"Error generating contextual summary: {e}")
            # Fallback to basic summary
            basic_summary = f"Episode {episode_id}: {title}\n\n"
            if scene_summaries:
                basic_summary += "Scene Summary:\n" + "\n".join(scene_summaries[:5])
            return basic_summary
    
    def extract(self, content: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Extract/generate summary from content (implements BaseExtractor interface).
        
        Args:
            content: Content to summarize
            context: Additional context
            
        Returns:
            Generated summary
        """
        summary_type = context.get("type", "general") if context else "general"
        
        if summary_type == "episode":
            episode_id = context.get("episode_id")
            return self.generate_episode_summary(episode_id) if episode_id else "No episode ID provided."
        elif summary_type == "character":
            character_name = context.get("character_name")
            return self.generate_character_profile(character_name) if character_name else "No character name provided."
        else:
            # General content summarization
            system_prompt = """You are summarizing content from a TV show. Create a concise but comprehensive summary that captures the key points, character interactions, and plot developments."""
            
            user_prompt = f"Summarize this content:\n\n{content}"
            
            try:
                return self._call_llm(system_prompt, user_prompt)
            except Exception as e:
                logger.error(f"Error generating summary: {e}")
                return "Summary generation failed."

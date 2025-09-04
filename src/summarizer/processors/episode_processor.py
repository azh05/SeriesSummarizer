"""Episode processing pipeline."""

import logging
from typing import Dict, Any, List, Optional
from tqdm import tqdm

from ..models import Episode, EpisodeInfo
from ..database import ChromaDBManager
from ..extractors import SceneSegmenter, CharacterExtractor, RelationshipExtractor, PlotEventExtractor


logger = logging.getLogger(__name__)


class EpisodeProcessor:
    """Processes episodes through the complete analysis pipeline."""
    
    def __init__(self, 
                 db_manager: ChromaDBManager,
                 scene_segmenter: Optional[SceneSegmenter] = None,
                 character_extractor: Optional[CharacterExtractor] = None,
                 relationship_extractor: Optional[RelationshipExtractor] = None,
                 plot_event_extractor: Optional[PlotEventExtractor] = None):
        """Initialize episode processor.
        
        Args:
            db_manager: ChromaDB manager instance
            scene_segmenter: Scene segmentation extractor
            character_extractor: Character information extractor
            relationship_extractor: Relationship extractor
            plot_event_extractor: Plot event extractor
        """
        self.db_manager = db_manager
        
        # Initialize extractors with defaults if not provided
        self.scene_segmenter = scene_segmenter or SceneSegmenter()
        self.character_extractor = character_extractor or CharacterExtractor()
        self.relationship_extractor = relationship_extractor or RelationshipExtractor()
        self.plot_event_extractor = plot_event_extractor or PlotEventExtractor()
        
        logger.info("Episode processor initialized")
    
    def process_episode(self, transcript: str, episode_info: Dict[str, Any]) -> Episode:
        """Process a complete episode through the analysis pipeline.
        
        Args:
            transcript: Raw episode transcript
            episode_info: Episode metadata (season, episode, title, etc.)
            
        Returns:
            Processed Episode object
        """
        # Create episode info object
        ep_info = EpisodeInfo(**episode_info)
        episode_id = f"S{ep_info.season:02d}E{ep_info.episode:02d}"
        
        logger.info(f"Processing episode {episode_id}: {ep_info.title}")
        
        # Create episode object
        episode = Episode(
            id=episode_id,
            info=ep_info,
            transcript=transcript
        )
        
        # Check if episode already exists
        existing_episode = self.db_manager.get_episode_by_id(episode_id)
        if existing_episode:
            logger.warning(f"Episode {episode_id} already exists. Updating...")
            self.db_manager.delete_episode(episode_id)
        
        # Process episode through pipeline
        with tqdm(total=6, desc=f"Processing {episode_id}") as pbar:
            
            # Step 1: Segment into scenes
            pbar.set_description(f"{episode_id}: Segmenting scenes")
            scenes = self.scene_segmenter.extract(
                transcript, 
                context={"episode_id": episode_id}
            )
            pbar.update(1)
            
            # Step 2: Extract characters from each scene
            pbar.set_description(f"{episode_id}: Extracting characters")
            all_characters = []
            existing_character_names = set()
            
            for scene in scenes:
                characters = self.character_extractor.extract(
                    scene.content,
                    context={
                        "episode_id": episode_id,
                        "existing_characters": list(existing_character_names)
                    }
                )
                
                for char in characters:
                    if char.name not in existing_character_names:
                        all_characters.append(char)
                        existing_character_names.add(char.name)
                        episode.add_character(char.name)
            pbar.update(1)
            
            # Step 3: Extract relationships from each scene
            pbar.set_description(f"{episode_id}: Extracting relationships")
            all_relationships = []
            existing_relationship_ids = set()
            
            for scene in scenes:
                relationships = self.relationship_extractor.extract(
                    scene.content,
                    context={
                        "episode_id": episode_id,
                        "characters_present": scene.characters_present
                    }
                )
                
                for rel in relationships:
                    if rel.relationship_id not in existing_relationship_ids:
                        all_relationships.append(rel)
                        existing_relationship_ids.add(rel.relationship_id)
            pbar.update(1)
            
            # Step 4: Extract plot events from each scene
            pbar.set_description(f"{episode_id}: Extracting plot events")
            all_plot_events = []
            
            for scene in scenes:
                plot_events = self.plot_event_extractor.extract(
                    scene.content,
                    context={
                        "episode_id": episode_id,
                        "scene_id": scene.scene_id,
                        "characters_present": scene.characters_present
                    }
                )
                
                all_plot_events.extend(plot_events)
                
                # Add plot events to scene
                for event in plot_events:
                    scene.add_plot_event(event.id)
            pbar.update(1)
            
            # Step 5: Store everything in database
            pbar.set_description(f"{episode_id}: Storing in database")
            
            # Store characters
            for character in all_characters:
                try:
                    self.db_manager.add_character(character)
                except Exception as e:
                    logger.error(f"Error storing character {character.name}: {e}")
            
            # Store relationships
            for relationship in all_relationships:
                try:
                    self.db_manager.add_relationship(relationship)
                except Exception as e:
                    logger.error(f"Error storing relationship {relationship.relationship_id}: {e}")
            
            # Store plot events
            for plot_event in all_plot_events:
                try:
                    self.db_manager.add_plot_event(plot_event)
                except Exception as e:
                    logger.error(f"Error storing plot event {plot_event.id}: {e}")
            
            # Store scenes
            for scene in scenes:
                try:
                    self.db_manager.add_scene(scene)
                    episode.add_scene(scene.scene_id)
                except Exception as e:
                    logger.error(f"Error storing scene {scene.scene_id}: {e}")
            pbar.update(1)
            
            # Step 6: Generate episode summary
            pbar.set_description(f"{episode_id}: Generating summary")
            episode.summary = self._generate_episode_summary(episode, scenes, all_plot_events)
            
            # Calculate importance score
            episode.importance_score = self._calculate_episode_importance(scenes, all_plot_events)
            
            # Store episode
            try:
                self.db_manager.add_episode(episode)
            except Exception as e:
                logger.error(f"Error storing episode {episode_id}: {e}")
            pbar.update(1)
        
        logger.info(f"Successfully processed episode {episode_id}")
        logger.info(f"  - {len(scenes)} scenes")
        logger.info(f"  - {len(all_characters)} new characters")
        logger.info(f"  - {len(all_relationships)} new relationships")
        logger.info(f"  - {len(all_plot_events)} plot events")
        
        return episode
    
    def update_character_from_episode(self, character_name: str, episode_id: str) -> None:
        """Update an existing character with new information from an episode.
        
        Args:
            character_name: Name of character to update
            episode_id: Episode ID to analyze for updates
        """
        # Get episode scenes
        scenes = self.db_manager.get_scenes_for_episode(episode_id)
        
        # Get existing character
        existing_char_data = self.db_manager.get_character_by_name(character_name)
        if not existing_char_data:
            logger.warning(f"Character {character_name} not found for update")
            return
        
        # Extract updates from each scene where character appears
        all_updates = []
        for scene_data in scenes:
            scene_metadata = scene_data["metadata"]
            if character_name in scene_metadata.get("characters_present", []):
                updates = self.character_extractor.extract_character_updates(
                    scene_data["document"],
                    character_name,
                    context={"episode_id": episode_id, "scene_id": scene_data["id"]}
                )
                if updates:
                    all_updates.append(updates)
        
        # Apply updates to character
        # This would require loading the character object and updating it
        # Implementation depends on how you want to handle character updates
        logger.info(f"Found {len(all_updates)} character updates for {character_name}")
    
    def _generate_episode_summary(self, episode: Episode, scenes: List, plot_events: List) -> str:
        """Generate a comprehensive episode summary.
        
        Args:
            episode: Episode object
            scenes: List of scenes
            plot_events: List of plot events
            
        Returns:
            Episode summary
        """
        # Combine scene summaries and plot events into comprehensive summary
        scene_summaries = [scene.summary for scene in scenes if scene.summary]
        major_events = [event.description for event in plot_events 
                       if event.importance.value in ["critical", "high"]]
        
        if not scene_summaries and not major_events:
            return f"Episode {episode.episode_id}: {episode.info.title}"
        
        summary_parts = []
        
        if scene_summaries:
            summary_parts.append("Scene Summary:\n" + "\n".join(f"- {s}" for s in scene_summaries[:5]))
        
        if major_events:
            summary_parts.append("Major Events:\n" + "\n".join(f"- {e}" for e in major_events[:5]))
        
        return "\n\n".join(summary_parts)
    
    def _calculate_episode_importance(self, scenes: List, plot_events: List) -> float:
        """Calculate overall episode importance score.
        
        Args:
            scenes: List of scenes
            plot_events: List of plot events
            
        Returns:
            Importance score (0.0-1.0)
        """
        if not scenes and not plot_events:
            return 0.5
        
        # Average scene importance
        scene_importance = sum(scene.importance_score for scene in scenes) / len(scenes) if scenes else 0.5
        
        # Weight by plot event significance
        if plot_events:
            critical_events = sum(1 for event in plot_events if event.importance.value == "critical")
            high_events = sum(1 for event in plot_events if event.importance.value == "high")
            
            event_weight = (critical_events * 1.0 + high_events * 0.8) / len(plot_events)
            event_importance = min(1.0, event_weight)
        else:
            event_importance = 0.5
        
        # Combine with higher weight on plot events
        return (scene_importance * 0.3 + event_importance * 0.7)

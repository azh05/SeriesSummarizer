"""ChromaDB manager for storing and retrieving TV series information."""

import os
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from ..models import Episode, Scene, Character, Relationship, PlotEvent


logger = logging.getLogger(__name__)


def _filter_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Filter out None values from metadata as ChromaDB doesn't accept them."""
    return {k: v for k, v in metadata.items() if v is not None}


class ChromaDBManager:
    """Manager for ChromaDB operations with TV series data."""
    
    def __init__(self, 
                 persist_directory: str = "./chroma_db",
                 series_name: str = "default_series",
                 embedding_function: Optional[Any] = None):
        """Initialize ChromaDB manager.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            series_name: Name of the TV series (used for collection naming)
            embedding_function: Custom embedding function (defaults to OpenAI)
        """
        self.persist_directory = persist_directory
        self.series_name = series_name.lower().replace(" ", "_")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Set up embedding function
        if embedding_function is None:
            # Try OpenAI embeddings first (if available), otherwise use default
            openai_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("CHROMA_OPENAI_API_KEY")
            groq_api_key = os.getenv("GROQ_API_KEY")
            
            if openai_api_key:
                # Use OpenAI embeddings if available
                self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=openai_api_key,
                    model_name="text-embedding-3-small"
                )
                logger.info("Using OpenAI embeddings for ChromaDB")
            elif groq_api_key:
                # Groq doesn't provide embeddings, so use ChromaDB's default
                logger.info("Groq API key found but Groq doesn't provide embeddings. Using ChromaDB default embeddings.")
                self.embedding_function = None  # Let ChromaDB use its default
            else:
                # Use default embedding function if no API key is provided
                logger.warning("No API keys found. Using ChromaDB default embedding function.")
                self.embedding_function = None  # Let ChromaDB use its default
        else:
            self.embedding_function = embedding_function
        
        # Initialize collections
        self._init_collections()
    
    def _init_collections(self) -> None:
        """Initialize all required collections."""
        collection_configs = {
            "episodes": "Full episode transcripts and summaries",
            "scenes": "Individual scene breakdowns",
            "characters": "Character profiles and development",
            "relationships": "Character relationships and evolution", 
            "plot_events": "Plot points and story arcs"
        }
        
        self.collections = {}
        for collection_type, description in collection_configs.items():
            collection_name = f"{self.series_name}_{collection_type}"
            try:
                if self.embedding_function is not None:
                    collection = self.client.get_collection(
                        name=collection_name,
                        embedding_function=self.embedding_function
                    )
                else:
                    collection = self.client.get_collection(name=collection_name)
                logger.info(f"Loaded existing collection: {collection_name}")
            except Exception as e:
                # Collection doesn't exist, create it
                logger.debug(f"Collection {collection_name} not found: {e}")
                if self.embedding_function is not None:
                    collection = self.client.create_collection(
                        name=collection_name,
                        embedding_function=self.embedding_function,
                        metadata={"description": description}
                    )
                else:
                    collection = self.client.create_collection(
                        name=collection_name,
                        metadata={"description": description}
                    )
                logger.info(f"Created new collection: {collection_name}")
            
            self.collections[collection_type] = collection
    
    def add_episode(self, episode: Episode) -> None:
        """Add an episode to the database.
        
        Args:
            episode: Episode object to add
        """
        try:
            # Prepare document and metadata
            document = f"Episode {episode.episode_id}: {episode.info.title}\n\n"
            document += f"Summary: {episode.summary or 'No summary available'}\n\n"
            document += f"Transcript: {episode.transcript}"
            
            metadata = {
                "episode_id": episode.episode_id,
                "season": episode.info.season,
                "episode": episode.info.episode,
                "title": episode.info.title,
                "air_date": episode.info.air_date,
                "importance_score": episode.importance_score,
                "processed_at": episode.processed_at.isoformat()
            }
            
            self.collections["episodes"].add(
                documents=[document],
                metadatas=[_filter_metadata(metadata)],
                ids=[episode.episode_id]
            )
            
            logger.info(f"Added episode {episode.episode_id} to database")
            
        except Exception as e:
            logger.error(f"Error adding episode {episode.episode_id}: {e}")
            raise
    
    def add_scene(self, scene: Scene) -> None:
        """Add a scene to the database.
        
        Args:
            scene: Scene object to add
        """
        try:
            # Prepare document
            document = f"Scene {scene.scene_id}\n"
            document += f"Episode: {scene.episode_id}\n"
            document += f"Location: {scene.location or 'Unknown'}\n"
            document += f"Characters: {', '.join(scene.characters_present)}\n\n"
            document += f"Summary: {scene.summary or 'No summary'}\n\n"
            document += f"Content: {scene.content}"
            
            metadata = {
                "scene_id": scene.scene_id,
                "episode_id": scene.episode_id,
                "scene_number": scene.scene_number,
                "location": scene.location,
                "characters_present": ", ".join(scene.characters_present),
                "plot_relevance": scene.plot_relevance,
                "importance_score": scene.importance_score,
                "processed_at": scene.processed_at.isoformat()
            }
            
            self.collections["scenes"].add(
                documents=[document],
                metadatas=[_filter_metadata(metadata)],
                ids=[scene.scene_id]
            )
            
            logger.info(f"Added scene {scene.scene_id} to database")
            
        except Exception as e:
            logger.error(f"Error adding scene {scene.scene_id}: {e}")
            raise
    
    def add_character(self, character: Character) -> None:
        """Add a character to the database.
        
        Args:
            character: Character object to add
        """
        try:
            # Prepare document
            document = f"Character: {character.name}\n"
            if character.aliases:
                document += f"Aliases: {', '.join(character.aliases)}\n"
            document += f"Role: {character.role.value}\n"
            document += f"Description: {character.description or 'No description'}\n"
            document += f"Occupation: {character.occupation or 'Unknown'}\n"
            document += f"Background: {character.background or 'No background'}\n\n"
            
            document += f"Personality Traits: {', '.join(character.personality_traits)}\n"
            document += f"Goals/Motivations: {', '.join(character.goals_motivations)}\n"
            document += f"Skills/Abilities: {', '.join(character.skills_abilities)}\n"
            document += f"Character Arc: {character.character_arc or 'No defined arc'}\n\n"
            
            if character.important_quotes:
                document += f"Important Quotes:\n" + "\n".join(f"- {quote}" for quote in character.important_quotes)
            
            metadata = {
                "character_name": character.name,
                "role": character.role.value,
                "first_appearance": character.first_appearance,
                "last_appearance": character.last_appearance,
                "importance_score": character.importance_score,
                "created_at": character.created_at.isoformat(),
                "updated_at": character.updated_at.isoformat()
            }
            
            self.collections["characters"].add(
                documents=[document],
                metadatas=[_filter_metadata(metadata)],
                ids=[character.name.lower().replace(" ", "_")]
            )
            
            logger.info(f"Added character {character.name} to database")
            
        except Exception as e:
            logger.error(f"Error adding character {character.name}: {e}")
            raise
    
    def add_relationship(self, relationship: Relationship) -> None:
        """Add a relationship to the database.
        
        Args:
            relationship: Relationship object to add
        """
        try:
            # Prepare document
            document = f"Relationship: {relationship.character1} and {relationship.character2}\n"
            document += f"Type: {relationship.relationship_type.value}\n"
            document += f"Status: {relationship.current_status.value}\n"
            document += f"Description: {relationship.description or 'No description'}\n\n"
            
            if relationship.how_they_met:
                document += f"How they met: {relationship.how_they_met}\n"
            if relationship.relationship_dynamic:
                document += f"Dynamic: {relationship.relationship_dynamic}\n\n"
            
            if relationship.important_dialogue:
                document += f"Important Dialogue:\n" + "\n".join(f"- {dialogue}" for dialogue in relationship.important_dialogue)
            
            # Add relationship timeline
            if relationship.relationship_changes:
                document += f"\n\nRelationship Timeline:\n"
                for change in relationship.get_relationship_timeline():
                    document += f"- {change.episode_id}: {change.description}\n"
            
            metadata = {
                "relationship_id": relationship.relationship_id,
                "character1": relationship.character1,
                "character2": relationship.character2,
                "relationship_type": relationship.relationship_type.value,
                "current_status": relationship.current_status.value,
                "first_interaction": relationship.first_interaction,
                "importance_score": relationship.importance_score,
                "emotional_intensity": relationship.emotional_intensity,
                "created_at": relationship.created_at.isoformat(),
                "updated_at": relationship.updated_at.isoformat()
            }
            
            self.collections["relationships"].add(
                documents=[document],
                metadatas=[_filter_metadata(metadata)],
                ids=[relationship.relationship_id]
            )
            
            logger.info(f"Added relationship {relationship.relationship_id} to database")
            
        except Exception as e:
            logger.error(f"Error adding relationship {relationship.relationship_id}: {e}")
            raise
    
    def add_plot_event(self, plot_event: PlotEvent) -> None:
        """Add a plot event to the database.
        
        Args:
            plot_event: PlotEvent object to add
        """
        try:
            # Prepare document
            document = f"Plot Event: {plot_event.title}\n"
            document += f"Type: {plot_event.event_type.value}\n"
            document += f"Importance: {plot_event.importance.value}\n"
            document += f"Episode: {plot_event.episode_id}\n"
            if plot_event.scene_id:
                document += f"Scene: {plot_event.scene_id}\n"
            document += f"\nDescription: {plot_event.description}\n\n"
            
            if plot_event.characters_involved:
                document += f"Characters Involved: {', '.join(plot_event.characters_involved)}\n"
            if plot_event.plot_arc:
                document += f"Plot Arc: {plot_event.plot_arc}\n"
            if plot_event.themes:
                document += f"Themes: {', '.join(plot_event.themes)}\n\n"
            
            if plot_event.mystery_elements:
                document += f"Mystery Elements: {', '.join(plot_event.mystery_elements)}\n"
            if plot_event.reveals_information:
                document += f"Reveals: {', '.join(plot_event.reveals_information)}\n"
            if plot_event.questions_raised:
                document += f"Questions Raised: {', '.join(plot_event.questions_raised)}\n"
            if plot_event.questions_answered:
                document += f"Questions Answered: {', '.join(plot_event.questions_answered)}\n"
            
            metadata = {
                "event_id": plot_event.id,
                "title": plot_event.title,
                "event_type": plot_event.event_type.value,
                "importance": plot_event.importance.value,
                "episode_id": plot_event.episode_id,
                "scene_id": plot_event.scene_id,
                "plot_arc": plot_event.plot_arc,
                "emotional_impact": plot_event.emotional_impact,
                "plot_significance": plot_event.plot_significance,
                "created_at": plot_event.created_at.isoformat(),
                "updated_at": plot_event.updated_at.isoformat()
            }
            
            self.collections["plot_events"].add(
                documents=[document],
                metadatas=[_filter_metadata(metadata)],
                ids=[plot_event.id]
            )
            
            logger.info(f"Added plot event {plot_event.id} to database")
            
        except Exception as e:
            logger.error(f"Error adding plot event {plot_event.id}: {e}")
            raise
    
    def query_episodes(self, 
                      query: str, 
                      n_results: int = 5,
                      where_filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query episodes collection.
        
        Args:
            query: Search query
            n_results: Number of results to return
            where_filter: Optional metadata filter
            
        Returns:
            Query results
        """
        return self.collections["episodes"].query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter
        )
    
    def query_scenes(self, 
                    query: str, 
                    n_results: int = 10,
                    where_filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query scenes collection.
        
        Args:
            query: Search query
            n_results: Number of results to return
            where_filter: Optional metadata filter
            
        Returns:
            Query results
        """
        return self.collections["scenes"].query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter
        )
    
    def query_characters(self, 
                        query: str, 
                        n_results: int = 10,
                        where_filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query characters collection.
        
        Args:
            query: Search query
            n_results: Number of results to return
            where_filter: Optional metadata filter
            
        Returns:
            Query results
        """
        return self.collections["characters"].query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter
        )
    
    def query_relationships(self, 
                           query: str, 
                           n_results: int = 10,
                           where_filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query relationships collection.
        
        Args:
            query: Search query
            n_results: Number of results to return
            where_filter: Optional metadata filter
            
        Returns:
            Query results
        """
        return self.collections["relationships"].query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter
        )
    
    def query_plot_events(self, 
                         query: str, 
                         n_results: int = 10,
                         where_filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query plot events collection.
        
        Args:
            query: Search query
            n_results: Number of results to return
            where_filter: Optional metadata filter
            
        Returns:
            Query results
        """
        return self.collections["plot_events"].query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter
        )
    
    def get_character_by_name(self, character_name: str) -> Optional[Dict[str, Any]]:
        """Get character by exact name match.
        
        Args:
            character_name: Name of the character
            
        Returns:
            Character data or None if not found
        """
        try:
            result = self.collections["characters"].get(
                ids=[character_name.lower().replace(" ", "_")]
            )
            if result["documents"]:
                return {
                    "document": result["documents"][0],
                    "metadata": result["metadatas"][0]
                }
            return None
        except Exception as e:
            logger.error(f"Error getting character {character_name}: {e}")
            return None
    
    def get_episode_by_id(self, episode_id: str) -> Optional[Dict[str, Any]]:
        """Get episode by ID.
        
        Args:
            episode_id: Episode ID (e.g., 'S01E01')
            
        Returns:
            Episode data or None if not found
        """
        try:
            result = self.collections["episodes"].get(ids=[episode_id])
            if result["documents"]:
                return {
                    "document": result["documents"][0],
                    "metadata": result["metadatas"][0]
                }
            return None
        except Exception as e:
            logger.error(f"Error getting episode {episode_id}: {e}")
            return None
    
    def get_scenes_for_episode(self, episode_id: str) -> List[Dict[str, Any]]:
        """Get all scenes for a specific episode.
        
        Args:
            episode_id: Episode ID
            
        Returns:
            List of scene data
        """
        try:
            result = self.collections["scenes"].get(
                where={"episode_id": episode_id}
            )
            
            scenes = []
            for i, doc in enumerate(result["documents"]):
                scenes.append({
                    "document": doc,
                    "metadata": result["metadatas"][i],
                    "id": result["ids"][i]
                })
            
            # Sort by scene number
            scenes.sort(key=lambda x: x["metadata"].get("scene_number", 0))
            return scenes
            
        except Exception as e:
            logger.error(f"Error getting scenes for episode {episode_id}: {e}")
            return []
    
    def update_character(self, character: Character) -> None:
        """Update an existing character.
        
        Args:
            character: Updated character object
        """
        try:
            # Delete existing entry
            character_id = character.name.lower().replace(" ", "_")
            self.collections["characters"].delete(ids=[character_id])
            
            # Add updated version
            self.add_character(character)
            
            logger.info(f"Updated character {character.name}")
            
        except Exception as e:
            logger.error(f"Error updating character {character.name}: {e}")
            raise
    
    def delete_episode(self, episode_id: str) -> None:
        """Delete an episode and all its scenes.
        
        Args:
            episode_id: Episode ID to delete
        """
        try:
            # Delete episode
            self.collections["episodes"].delete(ids=[episode_id])
            
            # Delete all scenes for this episode
            scenes = self.get_scenes_for_episode(episode_id)
            scene_ids = [scene["id"] for scene in scenes]
            if scene_ids:
                self.collections["scenes"].delete(ids=scene_ids)
            
            logger.info(f"Deleted episode {episode_id} and {len(scene_ids)} scenes")
            
        except Exception as e:
            logger.error(f"Error deleting episode {episode_id}: {e}")
            raise
    
    def reset_database(self) -> None:
        """Reset all collections (DELETE ALL DATA)."""
        try:
            for collection_name in self.collections.keys():
                full_name = f"{self.series_name}_{collection_name}"
                self.client.delete_collection(name=full_name)
            
            # Reinitialize collections
            self._init_collections()
            
            logger.warning("Database reset - all data deleted")
            
        except Exception as e:
            logger.error(f"Error resetting database: {e}")
            raise
    
    def get_collection_counts(self) -> Dict[str, int]:
        """Get count of items in each collection.
        
        Returns:
            Dictionary with collection names and counts
        """
        counts = {}
        for collection_type, collection in self.collections.items():
            try:
                count = collection.count()
                counts[collection_type] = count
            except Exception as e:
                logger.error(f"Error getting count for {collection_type}: {e}")
                counts[collection_type] = -1
        
        return counts

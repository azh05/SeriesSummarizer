"""Main TV Series Agent class that orchestrates all components."""

import logging
import os
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dotenv import load_dotenv

from .database import ChromaDBManager
from .processors import EpisodeProcessor
from .extractors import SceneSegmenter, CharacterExtractor, RelationshipExtractor, PlotEventExtractor
from .generators import SummaryGenerator
from .queries import QueryInterface
from .models import Episode, Character, Relationship, PlotEvent
from .utils import (
    retry_with_backoff, 
    handle_api_errors, 
    validate_episode_info, 
    validate_transcript,
    validate_series_name,
    validate_groq_key,
    validate_openai_key,  # Keep for backward compatibility
    TVSeriesAgentError,
    DatabaseError,
    ValidationError
)

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class TVSeriesAgent:
    """Main AI agent for processing TV series episode transcripts."""
    
    def __init__(self, 
                 series_name: str,
                 persist_directory: str = "./chroma_db",
                 model_name: str = "llama-3.1-8b-instant",
                 temperature: float = 0.1,
                 validate_api_key: bool = True):
        """Initialize the TV Series Agent.
        
        Args:
            series_name: Name of the TV series
            persist_directory: Directory to persist ChromaDB data
            model_name: Groq model to use for analysis (via OpenAI client)
            temperature: Temperature for LLM generation
            validate_api_key: Whether to validate Groq API key on initialization
            
        Raises:
            ValidationError: If inputs are invalid
            TVSeriesAgentError: If initialization fails
        """
        # Validate inputs
        validate_series_name(series_name)
        
        if validate_api_key and not validate_groq_key():
            raise ValidationError("Invalid or missing Groq API key. Set GROQ_API_KEY environment variable.")
        
        self.series_name = series_name
        self.persist_directory = persist_directory
        self.model_name = model_name
        self.temperature = temperature
        
        try:
            # Initialize database manager
            self.db_manager = ChromaDBManager(
                persist_directory=persist_directory,
                series_name=series_name
            )
            
            # Initialize extractors
            extractor_kwargs = {
                "model_name": model_name,
                "temperature": temperature
            }
            
            self.scene_segmenter = SceneSegmenter(**extractor_kwargs)
            self.character_extractor = CharacterExtractor(**extractor_kwargs)
            self.relationship_extractor = RelationshipExtractor(**extractor_kwargs)
            self.plot_event_extractor = PlotEventExtractor(**extractor_kwargs)
            
            # Initialize processors and generators
            self.episode_processor = EpisodeProcessor(
                db_manager=self.db_manager,
                scene_segmenter=self.scene_segmenter,
                character_extractor=self.character_extractor,
                relationship_extractor=self.relationship_extractor,
                plot_event_extractor=self.plot_event_extractor
            )
            
            self.summary_generator = SummaryGenerator(self.db_manager, **extractor_kwargs)
            self.query_interface = QueryInterface(self.db_manager, self.summary_generator)
            
            logger.info(f"TV Series Agent initialized for '{series_name}'")
            logger.info(f"Database location: {persist_directory}")
            logger.info(f"Model: {model_name} (temperature: {temperature})")
            
            # Log collection counts
            counts = self.db_manager.get_collection_counts()
            logger.info(f"Existing data: {counts}")
            
        except Exception as e:
            logger.error(f"Failed to initialize TV Series Agent: {e}")
            raise TVSeriesAgentError(f"Initialization failed: {e}") from e
    
    @handle_api_errors
    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
    def process_episode(self, 
                       transcript: str, 
                       episode_info: Dict[str, Any]) -> Episode:
        """Process a complete episode through the analysis pipeline.
        
        Args:
            transcript: Raw episode transcript
            episode_info: Episode metadata (season, episode, title, air_date, etc.)
            
        Returns:
            Processed Episode object
            
        Raises:
            ValidationError: If inputs are invalid
            TVSeriesAgentError: If processing fails
        """
        try:
            # Validate inputs
            validate_transcript(transcript)
            validate_episode_info(episode_info)
            
            logger.info(f"Processing episode: S{episode_info['season']:02d}E{episode_info['episode']:02d} - {episode_info['title']}")
            
            # Process episode
            episode = self.episode_processor.process_episode(transcript, episode_info)
            
            logger.info(f"Successfully processed episode {episode.episode_id}")
            return episode
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to process episode: {e}")
            raise TVSeriesAgentError(f"Episode processing failed: {e}") from e
    
    def generate_episode_summary(self, season: int, episode: int) -> str:
        """Generate comprehensive episode summary.
        
        Args:
            season: Season number
            episode: Episode number
            
        Returns:
            Episode summary
        """
        episode_id = f"S{season:02d}E{episode:02d}"
        return self.summary_generator.generate_episode_summary(episode_id)
    
    def get_character_profile(self, character_name: str) -> Dict[str, Any]:
        """Get comprehensive character profile.
        
        Args:
            character_name: Name of the character
            
        Returns:
            Character profile data
        """
        return self.query_interface.get_character_profile(character_name)
    
    def get_relationship_history(self, character1: str, character2: str) -> Dict[str, Any]:
        """Get relationship history between two characters.
        
        Args:
            character1: First character name
            character2: Second character name
            
        Returns:
            Relationship history data
        """
        return self.query_interface.get_relationship_history(character1, character2)
    
    def get_plot_arc_summary(self, arc_name: str) -> Dict[str, Any]:
        """Get summary of a specific plot arc.
        
        Args:
            arc_name: Name of the plot arc
            
        Returns:
            Plot arc summary data
        """
        return self.query_interface.get_plot_arc_summary(arc_name)
    
    def find_scene(self, description: str, n_results: int = 5) -> Dict[str, Any]:
        """Find scenes matching a description using semantic search.
        
        Args:
            description: Description of what to search for
            n_results: Number of results to return
            
        Returns:
            Search results
        """
        return self.query_interface.find_scene(description, n_results)
    
    def get_episode_context(self, season: int, episode: int) -> Dict[str, Any]:
        """Get what the agent knows before a specific episode.
        
        Args:
            season: Season number
            episode: Episode number
            
        Returns:
            Context information available before this episode
        """
        return self.query_interface.get_episode_context(season, episode)
    
    def track_mystery(self, mystery_description: str) -> Dict[str, Any]:
        """Track clues and resolution of a mystery.
        
        Args:
            mystery_description: Description of the mystery to track
            
        Returns:
            Mystery tracking information
        """
        return self.query_interface.track_mystery(mystery_description)
    
    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Search across all collections.
        
        Args:
            query: Search query
            n_results: Number of results per collection
            
        Returns:
            Combined search results
        """
        return self.query_interface.search_all_collections(query, n_results)
    
    def get_series_statistics(self) -> Dict[str, Any]:
        """Get statistics about the processed series.
        
        Returns:
            Series statistics
        """
        try:
            counts = self.db_manager.get_collection_counts()
            
            # Get additional statistics
            stats = {
                "series_name": self.series_name,
                "total_episodes": counts.get("episodes", 0),
                "total_scenes": counts.get("scenes", 0),
                "total_characters": counts.get("characters", 0),
                "total_relationships": counts.get("relationships", 0),
                "total_plot_events": counts.get("plot_events", 0),
                "database_location": self.persist_directory,
                "last_updated": datetime.now().isoformat()
            }
            
            # Get episode range if we have episodes
            if counts.get("episodes", 0) > 0:
                try:
                    # This would require querying all episodes to get range
                    # For now, just indicate we have episodes
                    stats["has_episodes"] = True
                except:
                    stats["has_episodes"] = False
            else:
                stats["has_episodes"] = False
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting series statistics: {e}")
            return {"error": f"Failed to get statistics: {e}"}
    
    def export_character_data(self, character_name: str) -> Dict[str, Any]:
        """Export all data for a specific character.
        
        Args:
            character_name: Name of the character to export
            
        Returns:
            Complete character data
        """
        try:
            # Get character profile
            profile = self.get_character_profile(character_name)
            
            # Get all relationships
            relationships = self.query_interface.get_character_relationships(character_name)
            
            # Get scenes where character appears
            # Use semantic search instead of metadata filtering since ChromaDB doesn't support $contains
            scenes = self.db_manager.query_scenes(
                f"scenes with {character_name}",
                n_results=100
            )
            
            return {
                "character_name": character_name,
                "profile": profile,
                "relationships": relationships,
                "scenes": scenes,
                "export_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error exporting character data for {character_name}: {e}")
            return {"error": f"Failed to export character data: {e}"}
    
    def build_relationship_graph(self):
        """Build a NetworkX graph of character relationships.
        
        Returns:
            NetworkX graph object
        """
        return self.query_interface.build_relationship_graph()
    
    def reset_database(self, confirm: bool = False) -> bool:
        """Reset the entire database (DELETE ALL DATA).
        
        Args:
            confirm: Must be True to actually perform the reset
            
        Returns:
            True if reset was performed
            
        Warning:
            This will permanently delete all processed data!
        """
        if not confirm:
            logger.warning("Database reset not confirmed. Set confirm=True to actually reset.")
            return False
        
        try:
            logger.warning(f"Resetting database for series '{self.series_name}' - ALL DATA WILL BE DELETED")
            self.db_manager.reset_database()
            logger.warning("Database reset completed")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting database: {e}")
            raise DatabaseError(f"Database reset failed: {e}") from e
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check of all components.
        
        Returns:
            Health check results
        """
        health = {
            "timestamp": datetime.now().isoformat(),
            "series_name": self.series_name,
            "status": "healthy",
            "components": {}
        }
        
        try:
            # Check database connection
            try:
                counts = self.db_manager.get_collection_counts()
                health["components"]["database"] = {
                    "status": "healthy",
                    "collections": counts
                }
            except Exception as e:
                health["components"]["database"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health["status"] = "degraded"
            
            # Check Groq API
            try:
                if validate_groq_key():
                    health["components"]["groq_api"] = {"status": "healthy"}
                else:
                    health["components"]["groq_api"] = {"status": "unhealthy", "error": "Invalid API key"}
                    health["status"] = "degraded"
            except Exception as e:
                health["components"]["groq_api"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health["status"] = "degraded"
            
            # Check extractors
            try:
                # Simple test of extractors
                test_content = "This is a test scene with John and Mary talking."
                
                # Test scene segmenter
                scenes = self.scene_segmenter.extract(test_content, {"episode_id": "TEST"})
                health["components"]["scene_segmenter"] = {"status": "healthy", "test_scenes": len(scenes)}
                
            except Exception as e:
                health["components"]["extractors"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health["status"] = "degraded"
            
        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)
        
        return health
    
    def __str__(self) -> str:
        """String representation of the agent."""
        counts = self.db_manager.get_collection_counts()
        return f"TVSeriesAgent(series='{self.series_name}', episodes={counts.get('episodes', 0)}, characters={counts.get('characters', 0)})"
    
    def __repr__(self) -> str:
        """Detailed representation of the agent."""
        return f"TVSeriesAgent(series_name='{self.series_name}', persist_directory='{self.persist_directory}', model='{self.model_name}')"

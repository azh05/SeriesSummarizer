"""Query interface for retrieving information from the knowledge base."""

import logging
from typing import Dict, Any, List, Optional, Union
import networkx as nx

from ..database import ChromaDBManager
from ..generators import SummaryGenerator


logger = logging.getLogger(__name__)


class QueryInterface:
    """Interface for querying the TV series knowledge base."""
    
    def __init__(self, db_manager: ChromaDBManager, summary_generator: Optional[SummaryGenerator] = None):
        """Initialize query interface.
        
        Args:
            db_manager: ChromaDB manager instance
            summary_generator: Summary generator for creating profiles and summaries
        """
        self.db_manager = db_manager
        self.summary_generator = summary_generator or SummaryGenerator(db_manager)
    
    def get_character_profile(self, character_name: str) -> Dict[str, Any]:
        """Get comprehensive character profile.
        
        Args:
            character_name: Name of the character
            
        Returns:
            Character profile data
        """
        try:
            # Get character data from database
            character_data = self.db_manager.get_character_by_name(character_name)
            if not character_data:
                return {"error": f"Character '{character_name}' not found"}
            
            # Debug: Ensure character_data is a dict
            if not isinstance(character_data, dict):
                return {"error": f"Invalid character data format for '{character_name}'"}
            
            # Generate comprehensive profile
            profile_summary = self.summary_generator.generate_character_profile(character_name)
            
            # Get character's relationships
            relationships = self.get_character_relationships(character_name)
            
            # Debug: Ensure relationships is a list
            if not isinstance(relationships, list):
                relationships = []
            
            # Get character's key scenes
            # Use semantic search instead of metadata filtering since ChromaDB doesn't support $contains
            key_scenes = self.db_manager.query_scenes(
                f"important scenes with {character_name}",
                n_results=10
            )
            
            # Get character development timeline
            character_metadata = character_data["metadata"]
            
            return {
                "name": character_name,
                "profile_summary": profile_summary,
                "metadata": character_metadata,
                "relationships": relationships,
                "key_scenes": key_scenes,
                "first_appearance": character_metadata.get("first_appearance"),
                "total_appearances": len(key_scenes.get("documents", [])) if isinstance(key_scenes, dict) else 0,
                "importance_score": character_metadata.get("importance_score", 0.5)
            }
            
        except Exception as e:
            logger.error(f"Error getting character profile for {character_name}: {e}")
            return {"error": f"Error retrieving character profile: {e}"}
    
    def get_relationship_history(self, character1: str, character2: str) -> Dict[str, Any]:
        """Get relationship history between two characters.
        
        Args:
            character1: First character name
            character2: Second character name
            
        Returns:
            Relationship history data
        """
        try:
            # Generate relationship summary
            relationship_summary = self.summary_generator.generate_relationship_summary(character1, character2)
            
            # Get relationship data
            rel_id = f"{character1}_{character2}".replace(" ", "_").lower()
            alt_rel_id = f"{character2}_{character1}".replace(" ", "_").lower()
            
            rel_data = None
            try:
                rel_result = self.db_manager.collections["relationships"].get(ids=[rel_id])
                if rel_result["documents"]:
                    rel_data = {
                        "document": rel_result["documents"][0],
                        "metadata": rel_result["metadatas"][0]
                    }
            except:
                pass
            
            if not rel_data:
                try:
                    rel_result = self.db_manager.collections["relationships"].get(ids=[alt_rel_id])
                    if rel_result["documents"]:
                        rel_data = {
                            "document": rel_result["documents"][0],
                            "metadata": rel_result["metadatas"][0]
                        }
                except:
                    pass
            
            if not rel_data:
                return {"error": f"No relationship found between {character1} and {character2}"}
            
            # Get scenes where they interact
            # Use semantic search instead of metadata filtering since ChromaDB doesn't support $contains
            interaction_scenes = self.db_manager.query_scenes(
                f"{character1} and {character2} interact together",
                n_results=10
            )
            
            return {
                "character1": character1,
                "character2": character2,
                "relationship_summary": relationship_summary,
                "relationship_data": rel_data,
                "interaction_scenes": interaction_scenes,
                "relationship_type": rel_data["metadata"].get("relationship_type"),
                "current_status": rel_data["metadata"].get("current_status"),
                "first_interaction": rel_data["metadata"].get("first_interaction")
            }
            
        except Exception as e:
            logger.error(f"Error getting relationship history for {character1}-{character2}: {e}")
            return {"error": f"Error retrieving relationship history: {e}"}
    
    def get_plot_arc_summary(self, arc_name: str) -> Dict[str, Any]:
        """Get summary of a specific plot arc.
        
        Args:
            arc_name: Name of the plot arc
            
        Returns:
            Plot arc summary data
        """
        try:
            # Generate plot arc summary
            arc_summary = self.summary_generator.generate_plot_arc_summary(arc_name)
            
            # Get all events in this arc
            arc_events = self.db_manager.query_plot_events(
                f"plot arc {arc_name}",
                where_filter={"plot_arc": arc_name},
                n_results=50
            )
            
            # Get episodes involved in this arc
            involved_episodes = set()
            if arc_events.get("metadatas"):
                for metadata in arc_events["metadatas"]:
                    episode_id = metadata.get("episode_id")
                    if episode_id:
                        involved_episodes.add(episode_id)
            
            return {
                "arc_name": arc_name,
                "summary": arc_summary,
                "events": arc_events,
                "total_events": len(arc_events.get("documents", [])),
                "episodes_involved": sorted(list(involved_episodes)),
                "episode_count": len(involved_episodes)
            }
            
        except Exception as e:
            logger.error(f"Error getting plot arc summary for {arc_name}: {e}")
            return {"error": f"Error retrieving plot arc summary: {e}"}
    
    def find_scene(self, description: str, n_results: int = 5) -> Dict[str, Any]:
        """Find scenes matching a description using semantic search.
        
        Args:
            description: Description of what to search for
            n_results: Number of results to return
            
        Returns:
            Search results
        """
        try:
            # Search scenes
            scene_results = self.db_manager.query_scenes(description, n_results=n_results)
            
            # Format results
            formatted_results = []
            if isinstance(scene_results, dict) and scene_results.get("documents"):
                for i, doc in enumerate(scene_results["documents"]):
                    metadata = scene_results["metadatas"][i]
                    scene_id = scene_results["ids"][i]
                    distance = scene_results.get("distances", [None])[i] if scene_results.get("distances") else None
                    
                    formatted_results.append({
                        "scene_id": scene_id,
                        "episode_id": metadata.get("episode_id"),
                        "scene_number": metadata.get("scene_number"),
                        "summary": metadata.get("summary"),
                        "location": metadata.get("location"),
                        "characters": metadata.get("characters_present", []),
                        "importance_score": metadata.get("importance_score", 0.5),
                        "relevance_score": 1.0 - distance if distance is not None else None
                    })
            
            return {
                "query": description,
                "total_results": len(formatted_results),
                "results": formatted_results
            }
            
        except Exception as e:
            logger.error(f"Error finding scenes for '{description}': {e}")
            return {"error": f"Error searching scenes: {e}"}
    
    def get_episode_context(self, season: int, episode: int) -> Dict[str, Any]:
        """Get what the agent knows before a specific episode.
        
        Args:
            season: Season number
            episode: Episode number
            
        Returns:
            Context information available before this episode
        """
        try:
            episode_id = f"S{season:02d}E{episode:02d}"
            
            # Get all previous episodes
            previous_episodes = []
            for s in range(1, season + 1):
                max_ep = episode - 1 if s == season else 99  # Assume max 99 episodes per season
                for e in range(1, max_ep + 1):
                    prev_id = f"S{s:02d}E{e:02d}"
                    prev_episode = self.db_manager.get_episode_by_id(prev_id)
                    if prev_episode:
                        previous_episodes.append(prev_id)
            
            # Get characters known before this episode
            known_characters = set()
            character_introductions = {}
            
            # Get relationships established before this episode
            known_relationships = []
            
            # Get ongoing plot arcs
            active_plot_arcs = set()
            
            # Query for information from previous episodes
            if previous_episodes:
                # Get characters
                for ep_id in previous_episodes:
                    ep_data = self.db_manager.get_episode_by_id(ep_id)
                    if ep_data and ep_data["metadata"].get("characters_introduced"):
                        for char in ep_data["metadata"]["characters_introduced"]:
                            known_characters.add(char)
                            character_introductions[char] = ep_id
                    
                    if ep_data and ep_data["metadata"].get("plot_arcs"):
                        active_plot_arcs.update(ep_data["metadata"]["plot_arcs"])
                
                # Get relationships
                all_relationships = self.db_manager.query_relationships(
                    "all relationships",
                    n_results=100
                )
                
                if all_relationships.get("metadatas"):
                    for rel_meta in all_relationships["metadatas"]:
                        first_interaction = rel_meta.get("first_interaction")
                        if first_interaction and first_interaction in previous_episodes:
                            known_relationships.append({
                                "character1": rel_meta.get("character1"),
                                "character2": rel_meta.get("character2"),
                                "type": rel_meta.get("relationship_type"),
                                "established_in": first_interaction
                            })
            
            return {
                "target_episode": episode_id,
                "previous_episodes": previous_episodes,
                "known_characters": list(known_characters),
                "character_introductions": character_introductions,
                "known_relationships": known_relationships,
                "active_plot_arcs": list(active_plot_arcs),
                "total_previous_episodes": len(previous_episodes)
            }
            
        except Exception as e:
            logger.error(f"Error getting episode context for S{season:02d}E{episode:02d}: {e}")
            return {"error": f"Error retrieving episode context: {e}"}
    
    def track_mystery(self, mystery_description: str) -> Dict[str, Any]:
        """Track clues and resolution of a mystery.
        
        Args:
            mystery_description: Description of the mystery to track
            
        Returns:
            Mystery tracking information
        """
        try:
            # Search for mystery-related events
            mystery_events = self.db_manager.query_plot_events(
                mystery_description,
                n_results=20
            )
            
            # Filter for mystery-related events
            clues = []
            resolutions = []
            related_events = []
            
            if mystery_events.get("metadatas"):
                for i, metadata in enumerate(mystery_events["metadatas"]):
                    event_type = metadata.get("event_type", "")
                    doc = mystery_events["documents"][i]
                    
                    event_info = {
                        "episode_id": metadata.get("episode_id"),
                        "title": metadata.get("title"),
                        "description": doc,
                        "importance": metadata.get("importance"),
                        "mystery_elements": metadata.get("mystery_elements", [])
                    }
                    
                    if event_type == "mystery_clue":
                        clues.append(event_info)
                    elif event_type == "mystery_resolution":
                        resolutions.append(event_info)
                    elif metadata.get("mystery_elements"):
                        related_events.append(event_info)
            
            # Sort by episode
            clues.sort(key=lambda x: x["episode_id"])
            resolutions.sort(key=lambda x: x["episode_id"])
            related_events.sort(key=lambda x: x["episode_id"])
            
            return {
                "mystery": mystery_description,
                "clues": clues,
                "resolutions": resolutions,
                "related_events": related_events,
                "total_clues": len(clues),
                "total_resolutions": len(resolutions),
                "is_resolved": len(resolutions) > 0
            }
            
        except Exception as e:
            logger.error(f"Error tracking mystery '{mystery_description}': {e}")
            return {"error": f"Error tracking mystery: {e}"}
    
    def get_character_relationships(self, character_name: str) -> List[Dict[str, Any]]:
        """Get all relationships for a specific character.
        
        Args:
            character_name: Name of the character
            
        Returns:
            List of relationships
        """
        try:
            # Query relationships where character is involved
            relationships = self.db_manager.query_relationships(
                f"relationships with {character_name}",
                n_results=50
            )
            
            character_relationships = []
            if relationships.get("metadatas"):
                for i, metadata in enumerate(relationships["metadatas"]):
                    char1 = metadata.get("character1")
                    char2 = metadata.get("character2")
                    
                    # Check if this character is involved
                    if character_name in [char1, char2]:
                        other_character = char2 if char1 == character_name else char1
                        
                        character_relationships.append({
                            "other_character": other_character,
                            "relationship_type": metadata.get("relationship_type"),
                            "current_status": metadata.get("current_status"),
                            "first_interaction": metadata.get("first_interaction"),
                            "importance_score": metadata.get("importance_score", 0.5),
                            "emotional_intensity": metadata.get("emotional_intensity", 0.5)
                        })
            
            # Sort by importance
            character_relationships.sort(key=lambda x: x["importance_score"], reverse=True)
            
            return character_relationships
            
        except Exception as e:
            logger.error(f"Error getting relationships for {character_name}: {e}")
            return []
    
    def build_relationship_graph(self) -> nx.Graph:
        """Build a NetworkX graph of character relationships.
        
        Returns:
            NetworkX graph of relationships
        """
        try:
            G = nx.Graph()
            
            # Get all relationships
            all_relationships = self.db_manager.query_relationships("all relationships", n_results=1000)
            
            if all_relationships.get("metadatas"):
                for metadata in all_relationships["metadatas"]:
                    char1 = metadata.get("character1")
                    char2 = metadata.get("character2")
                    rel_type = metadata.get("relationship_type")
                    importance = metadata.get("importance_score", 0.5)
                    
                    if char1 and char2:
                        G.add_edge(char1, char2, 
                                 relationship_type=rel_type,
                                 importance=importance,
                                 weight=importance)
            
            return G
            
        except Exception as e:
            logger.error(f"Error building relationship graph: {e}")
            return nx.Graph()
    
    def search_all_collections(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Search across all collections.
        
        Args:
            query: Search query
            n_results: Number of results per collection
            
        Returns:
            Combined search results
        """
        try:
            results = {
                "query": query,
                "episodes": self.db_manager.query_episodes(query, n_results),
                "scenes": self.db_manager.query_scenes(query, n_results),
                "characters": self.db_manager.query_characters(query, n_results),
                "relationships": self.db_manager.query_relationships(query, n_results),
                "plot_events": self.db_manager.query_plot_events(query, n_results)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching all collections for '{query}': {e}")
            return {"error": f"Error performing search: {e}"}

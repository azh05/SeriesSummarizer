"""Data validation utilities."""

import logging
from typing import Dict, Any, List, Optional

from .error_handling import ValidationError


logger = logging.getLogger(__name__)


def validate_episode_info(episode_info: Dict[str, Any]) -> bool:
    """Validate episode information dictionary.
    
    Args:
        episode_info: Episode info dictionary
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    required_fields = ["season", "episode", "title"]
    
    # Check required fields
    for field in required_fields:
        if field not in episode_info:
            raise ValidationError(f"Missing required field: {field}")
        
        if episode_info[field] is None or episode_info[field] == "":
            raise ValidationError(f"Field '{field}' cannot be empty")
    
    # Validate data types
    if not isinstance(episode_info["season"], int) or episode_info["season"] < 1:
        raise ValidationError("Season must be a positive integer")
    
    if not isinstance(episode_info["episode"], int) or episode_info["episode"] < 1:
        raise ValidationError("Episode must be a positive integer")
    
    if not isinstance(episode_info["title"], str):
        raise ValidationError("Title must be a string")
    
    # Validate optional fields
    if "air_date" in episode_info and episode_info["air_date"] is not None:
        air_date = episode_info["air_date"]
        if not isinstance(air_date, str):
            raise ValidationError("Air date must be a string")
        
        # Basic date format validation (YYYY-MM-DD)
        import re
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', air_date):
            raise ValidationError("Air date must be in YYYY-MM-DD format")
    
    if "duration" in episode_info and episode_info["duration"] is not None:
        if not isinstance(episode_info["duration"], int) or episode_info["duration"] <= 0:
            raise ValidationError("Duration must be a positive integer (minutes)")
    
    logger.debug(f"Episode info validation passed for S{episode_info['season']:02d}E{episode_info['episode']:02d}")
    return True


def validate_transcript(transcript: str) -> bool:
    """Validate episode transcript.
    
    Args:
        transcript: Episode transcript text
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(transcript, str):
        raise ValidationError("Transcript must be a string")
    
    if not transcript or transcript.strip() == "":
        raise ValidationError("Transcript cannot be empty")
    
    # Check minimum length (at least 100 characters for a meaningful transcript)
    if len(transcript.strip()) < 100:
        raise ValidationError("Transcript too short (minimum 100 characters)")
    
    # Check maximum length (reasonable limit to avoid processing issues)
    max_length = 1_000_000  # 1MB of text
    if len(transcript) > max_length:
        raise ValidationError(f"Transcript too long (maximum {max_length:,} characters)")
    
    logger.debug(f"Transcript validation passed ({len(transcript):,} characters)")
    return True


def validate_series_name(series_name: str) -> bool:
    """Validate TV series name.
    
    Args:
        series_name: Name of the TV series
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(series_name, str):
        raise ValidationError("Series name must be a string")
    
    if not series_name or series_name.strip() == "":
        raise ValidationError("Series name cannot be empty")
    
    # Check for reasonable length
    if len(series_name.strip()) > 200:
        raise ValidationError("Series name too long (maximum 200 characters)")
    
    # Check for invalid characters that might cause database issues
    import re
    if not re.match(r'^[a-zA-Z0-9\s\-_\.\(\)\'\"]+$', series_name):
        raise ValidationError("Series name contains invalid characters")
    
    logger.debug(f"Series name validation passed: '{series_name}'")
    return True


def validate_character_name(character_name: str) -> bool:
    """Validate character name.
    
    Args:
        character_name: Character name
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(character_name, str):
        raise ValidationError("Character name must be a string")
    
    if not character_name or character_name.strip() == "":
        raise ValidationError("Character name cannot be empty")
    
    # Check for reasonable length
    if len(character_name.strip()) > 100:
        raise ValidationError("Character name too long (maximum 100 characters)")
    
    logger.debug(f"Character name validation passed: '{character_name}'")
    return True


def validate_database_config(config: Dict[str, Any]) -> bool:
    """Validate database configuration.
    
    Args:
        config: Database configuration dictionary
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(config, dict):
        raise ValidationError("Database config must be a dictionary")
    
    # Check required fields
    if "persist_directory" not in config:
        raise ValidationError("Missing persist_directory in database config")
    
    persist_dir = config["persist_directory"]
    if not isinstance(persist_dir, str) or not persist_dir.strip():
        raise ValidationError("persist_directory must be a non-empty string")
    
    # Validate path
    import os
    try:
        # Try to create directory if it doesn't exist
        os.makedirs(persist_dir, exist_ok=True)
        
        # Check if directory is writable
        test_file = os.path.join(persist_dir, ".write_test")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        
    except Exception as e:
        raise ValidationError(f"Cannot write to persist_directory '{persist_dir}': {e}")
    
    logger.debug(f"Database config validation passed: {config}")
    return True


def sanitize_input(text: str, max_length: Optional[int] = None) -> str:
    """Sanitize text input for processing.
    
    Args:
        text: Input text
        max_length: Maximum allowed length
        
    Returns:
        Sanitized text
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Remove null bytes and other problematic characters
    text = text.replace('\x00', '').replace('\r\n', '\n').replace('\r', '\n')
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    # Truncate if necessary
    if max_length and len(text) > max_length:
        text = text[:max_length].rsplit(' ', 1)[0] + '...'
        logger.warning(f"Text truncated to {max_length} characters")
    
    return text


def validate_search_query(query: str) -> bool:
    """Validate search query.
    
    Args:
        query: Search query string
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(query, str):
        raise ValidationError("Search query must be a string")
    
    if not query or query.strip() == "":
        raise ValidationError("Search query cannot be empty")
    
    # Check minimum length
    if len(query.strip()) < 2:
        raise ValidationError("Search query too short (minimum 2 characters)")
    
    # Check maximum length
    if len(query) > 1000:
        raise ValidationError("Search query too long (maximum 1000 characters)")
    
    logger.debug(f"Search query validation passed: '{query[:50]}...'")
    return True

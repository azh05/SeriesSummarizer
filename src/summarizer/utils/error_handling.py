"""Error handling utilities."""

import logging
import time
import functools
from typing import Any, Callable, Optional, Union
import openai
from requests.exceptions import RequestException


logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        exceptions: Tuple of exceptions to catch and retry
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries + 1} attempts: {e}")
                        raise e
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    
                    logger.warning(f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
            
            # This should never be reached, but just in case
            raise last_exception
        
        return wrapper
    return decorator


def handle_api_errors(func: Callable) -> Callable:
    """Decorator for handling common API errors.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function with error handling
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except openai.RateLimitError as e:
            logger.error(f"OpenAI rate limit exceeded in {func.__name__}: {e}")
            raise APIError(f"Rate limit exceeded: {e}") from e
        except openai.APIError as e:
            logger.error(f"OpenAI API error in {func.__name__}: {e}")
            raise APIError(f"API error: {e}") from e
        except RequestException as e:
            logger.error(f"Request error in {func.__name__}: {e}")
            raise APIError(f"Request error: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise
    
    return wrapper


class TVSeriesAgentError(Exception):
    """Base exception for TV Series Agent errors."""
    pass


class DatabaseError(TVSeriesAgentError):
    """Database operation error."""
    pass


class ExtractionError(TVSeriesAgentError):
    """Information extraction error."""
    pass


class ValidationError(TVSeriesAgentError):
    """Data validation error."""
    pass


class APIError(TVSeriesAgentError):
    """API-related error."""
    pass


def safe_execute(func: Callable, *args, default_return: Any = None, **kwargs) -> Any:
    """Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        *args: Function arguments
        default_return: Default value to return on error
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or default_return on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error executing {func.__name__}: {e}")
        return default_return


def validate_groq_key() -> bool:
    """Validate that GROQ API key is available and valid.
    
    Returns:
        True if key is valid, False otherwise
    """
    import os
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY environment variable not set")
        return False
    
    if not api_key.startswith("gsk_"):
        logger.error("Invalid Groq API key format")
        return False
    
    # Test API key with a simple request
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key
        )
        # Make a minimal request to test the key
        client.models.list()
        return True
    except Exception as e:
        logger.error(f"Groq API key validation failed: {e}")
        return False


# Keep the old function for backward compatibility
def validate_openai_key() -> bool:
    """Validate that OpenAI API key is available and valid.
    
    Deprecated: Use validate_groq_key() instead.
    
    Returns:
        True if key is valid, False otherwise
    """
    import os
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        return False
    
    if not api_key.startswith("sk-"):
        logger.error("Invalid OpenAI API key format")
        return False
    
    # Test API key with a simple request
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        # Make a minimal request to test the key
        client.models.list()
        return True
    except Exception as e:
        logger.error(f"OpenAI API key validation failed: {e}")
        return False

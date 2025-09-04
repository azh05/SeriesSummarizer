"""Utility for loading system prompts from text files."""

import os
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PromptLoader:
    """Loads and manages system prompts from text files."""
    
    def __init__(self, prompts_dir: Optional[str] = None):
        """Initialize the prompt loader.
        
        Args:
            prompts_dir: Directory containing prompt files. If None, uses default location.
        """
        if prompts_dir is None:
            # Default to prompts directory relative to this file
            current_dir = Path(__file__).parent.parent
            prompts_dir = current_dir / "prompts"
        
        self.prompts_dir = Path(prompts_dir)
        self._cache: Dict[str, str] = {}
        
        if not self.prompts_dir.exists():
            logger.warning(f"Prompts directory not found: {self.prompts_dir}")
    
    def load_prompt(self, prompt_name: str) -> str:
        """Load a prompt from file.
        
        Args:
            prompt_name: Name of the prompt file (without .txt extension)
            
        Returns:
            Prompt text content
            
        Raises:
            FileNotFoundError: If prompt file doesn't exist
        """
        # Check cache first
        if prompt_name in self._cache:
            return self._cache[prompt_name]
        
        prompt_file = self.prompts_dir / f"{prompt_name}.txt"
        
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # Cache the content
            self._cache[prompt_name] = content
            
            logger.debug(f"Loaded prompt: {prompt_name}")
            return content
            
        except Exception as e:
            logger.error(f"Error loading prompt {prompt_name}: {e}")
            raise
    
    def load_prompt_template(self, prompt_name: str, **kwargs) -> str:
        """Load a prompt template and format it with provided arguments.
        
        Args:
            prompt_name: Name of the prompt file (without .txt extension)
            **kwargs: Variables to format into the template
            
        Returns:
            Formatted prompt text
        """
        template = self.load_prompt(prompt_name)
        
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing template variable {e} for prompt {prompt_name}")
            raise
    
    def clear_cache(self):
        """Clear the prompt cache."""
        self._cache.clear()
    
    def list_available_prompts(self) -> list[str]:
        """List all available prompt files.
        
        Returns:
            List of prompt names (without .txt extension)
        """
        if not self.prompts_dir.exists():
            return []
        
        prompt_files = self.prompts_dir.glob("*.txt")
        return [f.stem for f in prompt_files]


# Global instance for convenient access
_default_loader = None


def get_prompt_loader() -> PromptLoader:
    """Get the default prompt loader instance."""
    global _default_loader
    if _default_loader is None:
        _default_loader = PromptLoader()
    return _default_loader


def load_prompt(prompt_name: str) -> str:
    """Convenience function to load a prompt using the default loader."""
    return get_prompt_loader().load_prompt(prompt_name)


def load_prompt_template(prompt_name: str, **kwargs) -> str:
    """Convenience function to load and format a prompt template."""
    return get_prompt_loader().load_prompt_template(prompt_name, **kwargs)

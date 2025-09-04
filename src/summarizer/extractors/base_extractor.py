"""Base extractor class with common functionality."""

import os
import logging
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

import openai


logger = logging.getLogger(__name__)


class BaseExtractor(ABC):
    """Base class for all extractors."""
    
    def __init__(self, 
                 model_name: str = "llama-3.1-8b-instant",
                 temperature: float = 0.1,
                 max_tokens: Optional[int] = None):
        """Initialize base extractor.
        
        Args:
            model_name: Groq model to use (via OpenAI client)
            temperature: Temperature for generation
            max_tokens: Maximum tokens for response
        """
        self.client = openai.OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ.get("GROQ_API_KEY")
        )
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the LLM with system and user prompts.
        
        Args:
            system_prompt: System message
            user_prompt: User message
            
        Returns:
            LLM response
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            content = response.choices[0].message.content
            logger.debug(f"LLM call completed - model: {self.model_name}, tokens: {response.usage.total_tokens}")
            return content
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            raise
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from LLM with enhanced error handling.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed JSON data
        """
        import json
        import re
        
        # Clean the response
        response = response.strip()
        
        # Try multiple extraction strategies
        json_str = None
        
        # Strategy 1: Extract from markdown code blocks
        code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL | re.IGNORECASE)
        if code_block_match:
            json_str = code_block_match.group(1)
        
        # Strategy 2: Find JSON object (improved regex)
        if not json_str:
            # Look for balanced braces
            brace_count = 0
            start_idx = None
            
            for i, char in enumerate(response):
                if char == '{':
                    if start_idx is None:
                        start_idx = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and start_idx is not None:
                        json_str = response[start_idx:i+1]
                        break
        
        # Strategy 3: Fallback to original regex
        if not json_str:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
        
        if not json_str:
            logger.error(f"No JSON found in response: {response[:500]}...")
            raise ValueError("No JSON found in response")
        
        # Try to parse JSON with error correction
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Initial JSON parse failed: {e}")
            logger.debug(f"Problematic JSON: {json_str}")
            
            # Try to fix common JSON issues
            corrected_json = self._fix_json_issues(json_str)
            
            try:
                result = json.loads(corrected_json)
                logger.info("Successfully parsed JSON after corrections")
                return result
            except json.JSONDecodeError as e2:
                logger.error(f"Failed to parse JSON even after corrections: {e2}")
                logger.error(f"Original response: {response}")
                logger.error(f"Extracted JSON: {json_str}")
                logger.error(f"Corrected JSON: {corrected_json}")
                raise ValueError(f"Could not parse JSON: {e2}")
    
    def _fix_json_issues(self, json_str: str) -> str:
        """Attempt to fix common JSON formatting issues.
        
        Args:
            json_str: Potentially malformed JSON string
            
        Returns:
            Corrected JSON string
        """
        import re
        
        # Remove comments
        json_str = re.sub(r'//.*$', '', json_str, flags=re.MULTILINE)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        
        # Fix duplicate keys by removing duplicates (keep last occurrence)
        lines = json_str.split('\n')
        seen_keys = set()
        cleaned_lines = []
        
        for line in lines:
            # Check if line contains a key definition
            key_match = re.match(r'\s*"([^"]+)"\s*:', line)
            if key_match:
                key = key_match.group(1)
                if key in seen_keys:
                    logger.debug(f"Removing duplicate key: {key}")
                    continue
                seen_keys.add(key)
            cleaned_lines.append(line)
        
        json_str = '\n'.join(cleaned_lines)
        
        # Fix missing commas (basic heuristic)
        json_str = re.sub(r'"\s*\n\s*"', '",\n  "', json_str)
        json_str = re.sub(r'}\s*\n\s*"', '},\n  "', json_str)
        json_str = re.sub(r']\s*\n\s*"', '],\n  "', json_str)
        
        # Fix trailing commas
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        return json_str
    
    def _normalize_data_types(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize data types to match expected Pydantic model fields.
        
        Args:
            data: Raw data dictionary
            
        Returns:
            Normalized data dictionary
        """
        if not isinstance(data, dict):
            return data
        
        normalized = {}
        
        for key, value in data.items():
            # Handle None strings
            if isinstance(value, str) and value.lower() in ['none', 'null', '']:
                normalized[key] = None
            # Convert single strings to lists for fields that expect lists
            elif key in ['foreshadowing', 'callbacks', 'characters_present', 'key_dialogue', 
                        'plot_events', 'character_developments', 'relationship_dynamics',
                        'emotional_tone', 'themes', 'aliases', 'personality_traits',
                        'skills_abilities', 'goals_motivations', 'fears_weaknesses',
                        'important_quotes', 'key_scenes', 'episode_appearances']:
                if isinstance(value, str):
                    # Split by common delimiters if it looks like a list
                    if any(delim in value for delim in [',', ';', '|']):
                        items = [item.strip() for item in re.split(r'[,;|]', value) if item.strip()]
                        normalized[key] = items
                    else:
                        normalized[key] = [value] if value else []
                elif isinstance(value, list):
                    normalized[key] = value
                else:
                    normalized[key] = []
            # Convert numbers to strings for age field
            elif key == 'age' and isinstance(value, (int, float)):
                normalized[key] = str(value)
            # Handle nested dictionaries
            elif isinstance(value, dict):
                normalized[key] = self._normalize_data_types(value)
            else:
                normalized[key] = value
        
        return normalized
    
    def _parse_list_response(self, response: str) -> List[str]:
        """Parse list response from LLM.
        
        Args:
            response: Raw LLM response
            
        Returns:
            List of items
        """
        import re
        
        # Split by newlines and clean up
        lines = response.strip().split('\n')
        items = []
        
        for line in lines:
            # Remove bullet points, numbers, etc.
            clean_line = re.sub(r'^[\s\-\*\d\.\)]+', '', line).strip()
            if clean_line and not clean_line.startswith('#'):
                items.append(clean_line)
        
        return items
    
    @abstractmethod
    def extract(self, content: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """Extract information from content.
        
        Args:
            content: Content to extract from
            context: Additional context information
            
        Returns:
            Extracted information
        """
        pass

# TV Series Summarizer Examples

This directory contains example scripts demonstrating how to use the TV Series Summarizer.

## Prerequisites

1. **OpenAI API Key**: Set your OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

2. **Install Dependencies**: Make sure you've installed all required dependencies:
   ```bash
   uv sync
   ```

## Examples

### Basic Usage (`basic_usage.py`)

Demonstrates the fundamental features:
- Processing a single episode
- Generating episode summaries
- Extracting character profiles
- Searching for scenes
- Getting series statistics

Run with:
```bash
python examples/basic_usage.py
```

### Advanced Usage (`advanced_usage.py`)

Shows more sophisticated features:
- Processing multiple episodes
- Tracking character development across episodes
- Analyzing relationships between characters
- Following plot arcs and mysteries
- Building relationship graphs
- Exporting character data

Run with:
```bash
python examples/advanced_usage.py
```

## Sample Output

The examples will create ChromaDB databases in the `examples/` directory:
- `example_chroma_db/` - Basic example database
- `advanced_chroma_db/` - Advanced example database

These directories contain the processed vector embeddings and metadata.

## Customization

You can modify the examples to:
- Use different OpenAI models (gpt-3.5-turbo vs gpt-4)
- Adjust temperature settings for different creativity levels
- Process your own TV show transcripts
- Experiment with different search queries

## Notes

- The examples use sample detective show transcripts
- Processing time depends on the OpenAI model used
- ChromaDB data persists between runs
- Use the `reset_database()` method to clear all data if needed

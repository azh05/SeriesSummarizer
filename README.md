# TV Series Summarizer

A comprehensive AI agent that processes TV series episode transcripts and maintains an intelligent knowledge base about the show using ChromaDB and Groq's LLM API. The system intelligently extracts, stores, and retrieves information about plot events, character development, and relationships.

## Features

### 🎬 Episode Processing
- **Scene Segmentation**: Automatically breaks episodes into individual scenes
- **Character Extraction**: Identifies and profiles characters with their traits, goals, and development
- **Relationship Analysis**: Tracks relationships between characters and their evolution
- **Plot Event Detection**: Extracts major plot points, mysteries, and story arcs
- **Comprehensive Summaries**: Generates detailed episode summaries with cross-references

### 🔍 Intelligent Querying
- **Character Profiles**: Get detailed character information and development over time
- **Relationship Histories**: Track how relationships evolve throughout the series
- **Plot Arc Summaries**: Follow specific storylines across multiple episodes
- **Semantic Search**: Find scenes and events using natural language descriptions
- **Mystery Tracking**: Follow clues and resolutions of ongoing mysteries

### 🗄️ Knowledge Base
- **Multi-Collection Storage**: Organized storage in ChromaDB with separate collections for episodes, scenes, characters, relationships, and plot events
- **Vector Embeddings**: Semantic search capabilities using ChromaDB embeddings (or OpenAI embeddings if available)
- **Metadata Rich**: Comprehensive metadata for efficient filtering and querying
- **Cross-Referencing**: Links between related information across collections

### 📊 Analysis & Insights
- **Character Journey Mapping**: Track character development arcs
- **Relationship Graphs**: Visualize character relationships using NetworkX
- **Continuity Checking**: Flag potential continuity errors
- **Theme Extraction**: Identify recurring themes and motifs
- **Statistics & Health Monitoring**: Comprehensive system health and usage statistics

## Installation

### Prerequisites
- Python 3.11+
- Groq API key (get one free at [console.groq.com](https://console.groq.com))
- UV package manager (recommended) or pip

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd SeriesSummarizer
   ```

2. **Install dependencies**:
   ```bash
   # Using uv (recommended)
   uv sync
   
   # Or using pip
   pip install -e .
   ```

3. **Set up environment variables**:
   ```bash
   export GROQ_API_KEY="your-groq-api-key"
   ```

## Quick Start

```python
from summarizer import TVSeriesAgent

# Initialize the agent
agent = TVSeriesAgent(
    series_name="My TV Show",
    persist_directory="./my_show_db",
    model_name="llama-3.1-8b-instant"
)

# Process an episode
episode_info = {
    "season": 1,
    "episode": 1,
    "title": "Pilot",
    "air_date": "2024-01-01"
}

episode = agent.process_episode(transcript, episode_info)

# Generate summaries and get insights
summary = agent.generate_episode_summary(1, 1)
character_profile = agent.get_character_profile("Main Character")
scene_results = agent.find_scene("dramatic confrontation")
```

## Core Components

### Data Models
- **Episode**: Complete episode data with metadata and summaries
- **Scene**: Individual scene breakdowns with analysis
- **Character**: Character profiles with development tracking
- **Relationship**: Character relationships and their evolution
- **PlotEvent**: Plot points and story elements

### Processing Pipeline
1. **Scene Segmentation**: Break transcript into logical scenes
2. **Information Extraction**: Extract characters, relationships, and plot events
3. **Database Storage**: Store with rich metadata in ChromaDB
4. **Cross-Referencing**: Link related information across collections
5. **Summary Generation**: Create comprehensive summaries

### Query Interface
- `get_character_profile(name)` - Character analysis
- `get_relationship_history(char1, char2)` - Relationship evolution
- `get_plot_arc_summary(arc)` - Plot arc tracking
- `find_scene(description)` - Semantic scene search
- `track_mystery(description)` - Mystery progression
- `search(query)` - Cross-collection search

## Examples

See the `examples/` directory for comprehensive usage examples:

- **Basic Usage**: Single episode processing and basic queries
- **Advanced Usage**: Multi-episode series with complex analysis

```bash
# Run basic example
python examples/basic_usage.py

# Run advanced example  
python examples/advanced_usage.py
```

## Architecture

```
src/summarizer/
├── agent.py              # Main TVSeriesAgent class
├── models/               # Pydantic data models
├── database/             # ChromaDB management
├── extractors/           # Information extraction modules
├── processors/           # Episode processing pipeline
├── generators/           # Summary and content generation
├── queries/              # Query interface
└── utils/                # Utilities and error handling
```

### ChromaDB Collections

- **episodes_collection**: Full episode transcripts and summaries
- **scenes_collection**: Individual scene breakdowns  
- **characters_collection**: Character profiles and development
- **relationships_collection**: Character relationships and evolution
- **plot_events_collection**: Major plot points and story arcs

## Configuration

### Model Selection
Choose between different OpenAI models based on your needs:

- **llama-3.1-8b-instant**: Fast, efficient analysis (free tier available)
- **gpt-3.5-turbo**: Faster processing (lower cost, good quality)

### Temperature Settings
- **0.1**: More focused, consistent analysis (recommended)
- **0.5**: Balanced creativity and consistency
- **0.9**: More creative but less consistent

## Error Handling

The system includes comprehensive error handling:
- **Retry Logic**: Automatic retries with exponential backoff
- **API Rate Limiting**: Handles OpenAI API limits gracefully
- **Validation**: Input validation for all data
- **Health Checks**: System health monitoring

## Performance Considerations

### Processing Speed
- Scene segmentation: ~30 seconds per episode
- Character extraction: ~1-2 minutes per episode
- Full episode processing: ~3-5 minutes per episode

### Cost Optimization
- Use gpt-3.5-turbo for initial processing
- Use llama-3.1-8b-instant for fast analysis and summaries
- Batch similar operations to reduce API calls

### Storage
- ChromaDB storage: ~10-50MB per episode depending on length
- Embeddings are cached for efficient querying
- Metadata is optimized for fast filtering

## Advanced Features

### Relationship Graph Analysis
```python
import networkx as nx

# Build relationship graph
graph = agent.build_relationship_graph()

# Analyze network properties
centrality = nx.degree_centrality(graph)
communities = nx.community.greedy_modularity_communities(graph)
```

### Custom Extractors
Extend the system with custom extractors:
```python
from summarizer.extractors.base_extractor import BaseExtractor

class CustomExtractor(BaseExtractor):
    def extract(self, content, context=None):
        # Your custom extraction logic
        return extracted_data
```

### Data Export
Export processed data for external analysis:
```python
# Export character data
character_data = agent.export_character_data("Character Name")

# Export series statistics
stats = agent.get_series_statistics()
```

## API Reference

### TVSeriesAgent

**Constructor**:
```python
TVSeriesAgent(
    series_name: str,
    persist_directory: str = "./chroma_db",
    model_name: str = "llama-3.1-8b-instant",
    temperature: float = 0.1,
    validate_api_key: bool = True
)
```

**Main Methods**:
- `process_episode(transcript, episode_info)` - Process complete episode
- `generate_episode_summary(season, episode)` - Generate episode summary
- `get_character_profile(character_name)` - Get character profile
- `get_relationship_history(char1, char2)` - Get relationship history
- `find_scene(description, n_results=5)` - Semantic scene search
- `search(query, n_results=5)` - Search all collections
- `get_series_statistics()` - Get processing statistics
- `health_check()` - System health status

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions, issues, or feature requests:
1. Check the examples directory for usage patterns
2. Review the API documentation
3. Open an issue on GitHub

## Roadmap

- [ ] Support for additional LLM providers (Anthropic, local models)
- [ ] Web interface for episode management
- [ ] Integration with popular transcript sources
- [ ] Advanced visualization tools
- [ ] Multi-language support
- [ ] Real-time processing capabilities
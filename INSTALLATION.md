# Installation Guide

## Quick Start

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Set up OpenAI API key**:
   ```bash
   export GROQ_API_KEY="your-groq-api-key-here"
   ```

3. **Test the installation**:
   ```bash
   uv run python test_simple.py
   ```

4. **Run examples**:
   ```bash
   uv run python examples/basic_usage.py
   ```

## System Requirements

- Python 3.11+
- OpenAI API key (for full functionality)
- At least 2GB free disk space for ChromaDB storage

## Dependencies

The system uses the following key dependencies:
- **langchain** & **langchain-openai**: LLM integration
- **chromadb**: Vector database for semantic storage
- **pydantic**: Data validation and modeling
- **networkx**: Relationship graph analysis
- **tqdm**: Progress bars during processing

## Troubleshooting

### Import Errors
If you see import errors, make sure to run commands with `uv run`:
```bash
uv run python your_script.py
```

### ChromaDB Issues
ChromaDB will create collections automatically on first use. If you encounter collection errors, try:
1. Delete the ChromaDB directory (e.g., `./chroma_db`)
2. Restart your script

### API Key Issues
The system will work with default embeddings if no OpenAI API key is provided, but functionality will be limited. Set your API key with:
```bash
export GROQ_API_KEY="gsk_your-key-here"
```

### Performance
- First run may be slower as ChromaDB initializes
- Processing time depends on episode length and OpenAI model used
- Use `gpt-3.5-turbo` for faster/cheaper processing
- Use `gpt-4` for highest quality analysis

## Verification

Run the test scripts to verify everything is working:

```bash
# Basic functionality test
uv run python test_simple.py

# Full installation test (requires OpenAI API key)
uv run python test_installation.py

# Example usage
uv run python examples/basic_usage.py
```

## Next Steps

Once installed successfully:
1. Review the examples in the `examples/` directory
2. Read the main `README.md` for detailed usage instructions
3. Start processing your own TV show transcripts!

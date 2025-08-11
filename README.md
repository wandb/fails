# fails

A tool for analyzing evaluation failures and categorizing them automatically. The pipeline fetches evaluation data from Weave, categorizes failures, and clusters them into consistent failure categories.

## Setup

1. Install dependencies:
```bash
uv sync
```

2. Set up environment variables:
```bash
# Create .env file
touch .env
```

Add configuration to `.env`:

### Required: API Keys
Add the API key for your chosen LLM provider:
```env
# For Google/Gemini models
GOOGLE_API_KEY=your_google_api_key_here

# For Anthropic/Claude models  
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# For OpenAI models
OPENAI_API_KEY=your_openai_api_key_here
```

### Optional: Pipeline Configuration
```env
# Weave logging configuration (optional)
WANDB_LOGGING_ENTITY=your-team-name  # Optional: W&B entity for logging
WANDB_LOGGING_PROJECT=eval-failures   # Project name for Weave logging (default: eval-failures)

# Model configuration (optional)
MODEL=gemini/gemini-2.5-pro          # LLM model to use (default: gemini/gemini-2.5-pro)

# Processing configuration (optional)
N_SAMPLES=10                         # Number of samples to process (default: all)
MAX_CONCURRENT_LLM_CALLS=5           # Max concurrent LLM API calls (default: 20)
```

**Note:** Environment variables are overridden by CLI arguments if both are provided.

## Run Failure Categorization

Analyze evaluation failures and categorize them:

```bash
# Run with default model (Gemini 2.5 Pro)
uv run python fails/pipeline.py

# Debug mode (uses faster gemini-2.5-flash, limits to 5 samples)
uv run python fails/pipeline.py --debug

# Use different models
uv run python fails/pipeline.py --model "openai/gpt-4o"
uv run python fails/pipeline.py --model "anthropic/claude-3-5-sonnet-latest"
uv run python fails/pipeline.py --model "gemini/gemini-2.0-flash-exp"

# Limit samples and concurrency
uv run python fails/pipeline.py --n-samples 10 --max-concurrent-llm-calls 5
```

## Run Evals

Available evaluations in `evals/`:
- `speaker_classification/` - Classifies speakers as internal (W&B employees) or external (prospects/users)

Run speaker classification eval:
```bash
cd evals/speaker_classification
uv run python speaker_classification.py
```

### Supported Models

The tool uses [LiteLLM](https://docs.litellm.ai/docs/providers) format: `provider/model-name`, e.g. `gemini/gemini-2.5-pro`, `openai/gpt-5` etc
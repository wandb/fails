# fails

A tool for analyzing evaluation failures and categorizing them automatically.

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

Add required API keys to `.env`:
```env
GOOGLE_API_KEY=your_google_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here  
OPENAI_API_KEY=your_openai_api_key_here
```

## Run Evals

Available evaluations in `evals/`:
- `speaker_classification/` - Classifies speakers as internal (W&B employees) or external (prospects/users)

Run speaker classification eval:
```bash
cd evals/speaker_classification
uv run python speaker_classification.py
```

## Run Failure Categorization

Analyze evaluation failures and categorize them:

```bash
# Run failure categorization pipeline
uv run python fails/pipeline.py

# Debug mode (faster model, limited traces)
uv run python fails/pipeline.py --debug

# Use specific model
uv run python fails/pipeline.py --model "gemini/gemini-2.5-pro"
```

The pipeline fetches evaluation data from Weave, categorizes failures, and clusters them into consistent failure categories.
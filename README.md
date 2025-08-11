<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/wandb/wandb/main/assets/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/wandb/wandb/main/assets/logo-light.svg">
    <img src="https://raw.githubusercontent.com/wandb/wandb/main/assets/logo-light.svg" width="600" alt="Weights & Biases">
  </picture>
</p>


# FAILS

***Look at your most relevant failures first***

FAILS is a tool for reviewing evaluation failures and categorizing them. The pipeline fetches evaluation data from [Weave](https://weave-docs.wandb.ai/), categorizes failures, clusters them into failure categories and outputs a report with the failure clusters. The goal is not to remove the need to look at your data, but instead try and show the most impactful types of failures happening.

## Setup

1. Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Add your model name to .env in [LiteLLM](https://docs.litellm.ai/docs/providers) format: `provider/model-name`, e.g. `gemini/gemini-2.5-pro`, `openai/gpt-5` etc:

```bash
LLM_MODEL=gemini/gemini-2.5-pro
```

3. Add the API key for your chosen LLM provider to `.env`:
```env
# For Google/Gemini models
GOOGLE_API_KEY=your_google_api_key_here

# For Anthropic/Claude models  
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# For OpenAI models
OPENAI_API_KEY=your_openai_api_key_here

...
```

## Run Failure Categorization Pipeline

Analyze evaluation failures and categorize them. The pipeline will ask for a Weave Evaluation URL, some context about your task, and the relevant columns that have been logged to Weave to help with the failure categorization. 

```bash
# Run with default model (Gemini 2.5 Pro)
uv run fails/pipeline.py
```

## Additional options
You can adjust the pipeline settings either via the env vars in `.env` or via cli arguments:

```bash
# Debug mode (uses faster gemini-2.5-flash, limits to 5 samples)
uv run fails/pipeline.py --debug

# Use different models
uv run fails/pipeline.py --model "openai/gpt-5"
uv run fails/pipeline.py --model "anthropic/claude-opus-4-1-20250805"
uv run fails/pipeline.py --model "gemini/gemini-2.5-pro"

# Limit samples and concurrency
uv run fails/pipeline.py --n-samples 10 --max-concurrent-llm-calls 5
```

### Optional: Pipeline Configuration
Additional settings in the `.env` file:

```bash
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

## Run the pipeline eval

Available evaluations in `evals/`:
- `speaker_classification/` - Classifies speakers as internal (W&B employees) or external (prospects/users)

Run speaker classification eval:
```bash
cd evals/speaker_classification
uv run python speaker_classification.py
```

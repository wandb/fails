# Speaker Classification Eval

Classifies speakers in conversation transcripts as internal (W&B employees) or external (prospects/users).

## Setup

1. Download dataset:
```bash
uv run python download_weave_dataset.py
```

2. Ensure API keys are set in your `.env` file:
```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

## Run

```bash
uv run python speaker_classification.py
```

## How it works

- Downloads conversation transcripts from Weave dataset
- Uses LLM to analyze each speaker's lines and classify them as internal or external
- Defaults to external classification unless there's clear evidence of W&B employment
- Evaluates predictions against ground truth affiliation labels
- Results are logged to Weave project `wandb-applied-ai-team/eval-failures` 
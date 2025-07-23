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

## Evaluation System

The failure categorization system uses sophisticated dataset-level metrics that handle models **inventing new category names**. Traditional evaluation fails when models predict `"format_error"` while ground truth is `"Output-Label Format Error"` - our system recognizes these as conceptually equivalent.

### How It Works

#### 1. Hungarian Assignment Algorithm
- **Groups examples** by predicted/ground truth categories  
- **Calculates Jaccard overlap** between category clusters (not string similarity)
- **Finds optimal 1-to-1 mapping** using Hungarian algorithm
- **Example**: `"format_error"` → `"Org Reference Ambiguity"` because they cluster the same data points

#### 2. Adjusted Rand Index (ARI)
- **Measures partition agreement** ignoring label names
- **Range**: -1 (opposite) to +1 (perfect clustering)
- **Perfect for models that invent new names** but group correctly

#### 3. Category-Discovery F1  
- **Coverage**: % of ground truth categories discovered (60% in latest eval)
- **Precision**: % of predicted categories that are valid (100% in latest eval)  
- **F1**: Harmonic mean balancing discovery vs over-generation (75% in latest eval)

### Run Full Evaluation

```bash
# Run dataset-level evaluation with sophisticated scoring
cd evaluation/fails_eval  
uv run python failure_categorization_eval.py --run_eval --debug

# Results logged to Weights & Biases Weave UI
# View detailed analysis at: https://wandb.ai/wandb-applied-ai-team/eval-failures/weave
```

### Key Insight
Models get **100% precision** despite zero string matches because the evaluation measures **conceptual clustering behavior**, not exact label matching. This enables proper evaluation of AI systems that naturally invent their own vocabulary.
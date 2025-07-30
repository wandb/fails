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

## Meta Evaluation System

The meta evaluation system evaluates the failure categorization pipeline itself using sophisticated dataset-level metrics that handle models **inventing new category names**. Traditional evaluation fails when models predict `"format_error"` while ground truth is `"Output-Label Format Error"` - our system recognizes these as conceptually equivalent.

### How Meta Evaluations Work

#### 1. Real Pipeline Integration
- **Uses the complete 3-step pipeline**: Draft categorization → Clustering/Review → Final classification
- **No mock models**: Evaluates the actual production pipeline logic
- **Individual predictions**: Each dataset example runs through the full pipeline

#### 2. Sophisticated Dataset-Level Metrics

##### **Hungarian Assignment Algorithm**
- **Groups examples** by predicted/ground truth categories  
- **Calculates Jaccard overlap** between category clusters (not string similarity)
- **Finds optimal 1-to-1 mapping** using Hungarian algorithm
- **Example**: `"format_error"` → `"Org Reference Ambiguity"` because they cluster the same data points

##### **Adjusted Rand Index (ARI)**
- **Measures partition agreement** ignoring label names
- **Range**: -1 (opposite) to +1 (perfect clustering)
- **Perfect for models that invent new names** but group correctly

##### **Category-Discovery F1**  
- **Coverage**: % of ground truth categories discovered
- **Precision**: % of predicted categories that are valid  
- **F1**: Harmonic mean balancing discovery vs over-generation

### Execute Meta Evaluations

#### **Main Meta Evaluation** (evaluates the real pipeline)
```bash
# Run meta evaluation with real pipeline
uv run python -m evaluation.fails_eval.failure_categorization_eval --run_eval

# Debug mode (faster model, detailed output)
uv run python -m evaluation.fails_eval.failure_categorization_eval --run_eval --debug
```

#### **Pipeline Test** (simpler version)
```bash
# Run pipeline test directly
uv run python pipe_test.py
```

#### **Main Pipeline** (what gets evaluated)
```bash
# Run the actual failure categorization pipeline
uv run python fails/pipeline.py

# Debug mode
uv run python fails/pipeline.py --debug
```

### Meta Evaluation Architecture

```
Dataset Examples → Real Pipeline (per example) → Sophisticated Metrics
     ↓                    ↓                           ↓
Ground Truth      3-Step Process:              ARI + Category F1
Categories    1. Draft categorization            ↓
     ↓        2. Clustering/Review         Hungarian Assignment
     ↓        3. Final classification           ↓
     ↓                    ↓              Optimal Category Mapping
     └────── Comparison ──┘                     ↓
                                         Performance Scores
```

### Key Features

#### **Real Pipeline Evaluation**
- Uses the actual production pipeline code from `fails/pipeline.py`
- Each prediction runs the complete 3-step process
- No mocks or simplified versions

#### **Dataset-Level Intelligence**  
- **Handles invented categories**: Model can predict `"logic_error"` when truth is `"Reasoning Failure"`
- **Jaccard-based matching**: Maps categories by data point overlap, not string similarity
- **Partition evaluation**: Measures how well the model groups similar failures

#### **Results & Monitoring**
- Results logged to **Weights & Biases Weave UI**
- View at: https://wandb.ai/wandb-applied-ai-team/eval-failures/weave
- Dataset: `speaker_classification_failure_annotation:v0`
- Project: `wandb-applied-ai-team/eval-failures`

### Key Insight

Models get **high scores** despite zero string matches because the evaluation measures **conceptual clustering behavior**, not exact label matching. This enables proper evaluation of AI systems that naturally invent their own vocabulary for failure categories.

**Example**: Pipeline predicts `["format_error", "logic_error", "hallucination"]` while ground truth is `["Output Format Issue", "Reasoning Problem", "False Information"]` → **High ARI score** because the groupings are conceptually correct.

### Testing

```bash
# Run pipeline test (simpler test version)
uv run python pipe_test.py
```

### Setup Notes
- Requires Python 3.13+ (update .python-version if needed)
- Uses `uv` package manager for dependency management  
- Evaluation data pulled from W&B Weave project: `wandb-applied-ai-team/eval-failures`
- Meta evaluation uses dataset: `speaker_classification_failure_annotation:v0`
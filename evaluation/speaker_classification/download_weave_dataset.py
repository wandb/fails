import weave
import pandas as pd
from weave import Dataset
import os

# Always save to evals/datasets/speaker_classification.csv relative to this script
csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets', 'speaker_classification.csv'))

# Download from source project
download_project = 'agatamlyn/gong-debug'
weave.init(download_project)
dataset = weave.ref('Dataset:v5').get()
try:
    df = dataset.to_pandas()
    df.to_csv(csv_path, index=False)
    print(f'Dataset saved to {csv_path}')
except Exception as e:
    print(f'Error saving dataset: {e}')
    df = None

# Push to target project if download succeeded
if df is not None:
    target_project = 'wandb-applied-ai-team/eval-failures'
    weave.init(target_project)
    ds = Dataset.from_pandas(df)
    ds.name = 'speaker_classification'
    weave.publish(ds)
    print('Dataset pushed to Weave project wandb-applied-ai-team/eval-failures as speaker_classification') 
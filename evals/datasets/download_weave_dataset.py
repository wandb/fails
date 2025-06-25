import weave
import pandas as pd

# Initialize Weave with the project
weave.init('agatamlyn/gong-debug')

# Get the dataset reference
dataset = weave.ref('Dataset:v5').get()

# Convert to pandas DataFrame
# The Weave docs say to use .to_pandas() on a Dataset object
try:
    df = dataset.to_pandas()
    df.to_csv('evals/datasets/Dataset_v5.csv', index=False)
    print('Dataset saved to evals/datasets/Dataset_v5.csv')
except Exception as e:
    print(f'Error saving dataset: {e}') 
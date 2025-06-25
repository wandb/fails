import json
import os

def summarize_json(file_path):
    """
    Summarizes the structure of a large JSON file.
    """
    try:
        with open(file_path, 'r') as f:
            # For large files, it's better to not load the whole file in memory.
            # However, for summarizing structure, we might need to.
            # A better approach for very large files would be to read line by line if it's a JSONL file,
            # or use a library that can parse JSON incrementally.
            # For now, we assume it can be loaded into memory.
            data = json.load(f)

        if isinstance(data, list):
            print(f"The JSON is a list of {len(data)} elements.")
            if len(data) > 0:
                print("The first element is a dictionary with the following keys:")
                print(list(data[0].keys()))
                print("\nSample of the first element:")
                # pretty print the first element
                import pprint
                pprint.pprint(data[0])
        elif isinstance(data, dict):
            print("The JSON is a dictionary with the following keys:")
            print(list(data.keys()))
            # If the values are lists, print their lengths and a sample from each.
            for key, value in data.items():
                if isinstance(value, list):
                    print(f"  - Key '{key}' contains a list of {len(value)} elements.")
                    if len(value) > 0:
                        print("    Sample of the first element:")
                        import pprint
                        pprint.pprint(value[0])

        else:
            print(f"The JSON contains a {type(data)}.")

    except json.JSONDecodeError:
        print("The file is not a valid JSON file.")
    except MemoryError:
        print("File is too large to fit in memory. Consider using a streaming JSON parser.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    file_path = "evals/datasets/transcript_speaker_classification.json"
    if os.path.exists(file_path):
        summarize_json(file_path)
    else:
        print(f"File not found: {file_path}") 
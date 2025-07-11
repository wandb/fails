from typing import Any, Dict, List
import weave


def filter_dict_by_paths(data: Dict[str, Any], allowed_paths: set[str]) -> Dict[str, Any]:
    """Filter a nested dictionary to only include data from allowed paths."""
    result = {}
    
    # Group paths by their top-level key
    paths_by_top_level = {}
    for path in allowed_paths:
        parts = path.split('.', 1)
        top_level = parts[0]
        if top_level not in paths_by_top_level:
            paths_by_top_level[top_level] = []
        if len(parts) > 1:
            paths_by_top_level[top_level].append(parts[1])
        else:
            # This is a top-level field
            paths_by_top_level[top_level] = None
    
    # Process each top-level key
    for top_level, sub_paths in paths_by_top_level.items():
        if top_level not in data:
            continue
            
        if sub_paths is None:
            # Include the entire top-level value
            result[top_level] = data[top_level]
        else:
            # Filter nested structure
            result[top_level] = filter_nested_dict(data[top_level], sub_paths)
    
    return result


def filter_nested_dict(data: Any, allowed_sub_paths: List[str]) -> Any:
    """Recursively filter nested dictionary based on allowed sub-paths."""
    if not isinstance(data, dict):
        return data
    
    result = {}
    
    # Group sub-paths by their next level
    paths_by_next_level = {}
    direct_keys = set()
    
    for path in allowed_sub_paths:
        parts = path.split('.', 1)
        if len(parts) == 1:
            # Direct key at this level
            direct_keys.add(parts[0])
        else:
            # Nested path
            next_level = parts[0]
            if next_level not in paths_by_next_level:
                paths_by_next_level[next_level] = []
            paths_by_next_level[next_level].append(parts[1])
    
    # Include direct keys
    for key in direct_keys:
        if key in data:
            result[key] = data[key]
    
    # Recursively filter nested structures
    for key, sub_paths in paths_by_next_level.items():
        if key in data:
            result[key] = filter_nested_dict(data[key], sub_paths)
    
    return result if result else None


@weave.op
def filter_trace_data_by_columns(traces: List[Dict[str, Any]], selected_columns: List[str]) -> List[Dict[str, Any]]:
    """
    Filter trace data to only include the selected column paths.
    
    Args:
        traces: List of trace dictionaries
        selected_columns: List of column paths to keep (e.g., ["inputs.example.call_name"])
    
    Returns:
        List of filtered trace dictionaries
    """
    selected_paths = set(selected_columns)
    filtered_traces = []
    
    for trace in traces:
        filtered_trace = filter_dict_by_paths(trace, selected_paths)
        # Always preserve essential fields
        for key in ['id', 'trace_id', 'parent_id', 'display_name']:
            if key in trace and key not in filtered_trace:
                filtered_trace[key] = trace[key]
        filtered_traces.append(filtered_trace)
    
    return filtered_traces
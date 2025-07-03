# Weave Query API Documentation

This document outlines the key learnings from researching the wandb-mcp-server implementation of the Weave query API, focusing on how to query evaluation trace data.

## Overview

The Weave query API provides a way to programmatically access trace data from Weave evaluations. The API is RESTful and returns data in JSONL (JSON Lines) format for efficient streaming of large datasets.

## API Endpoint

- **Base URL**: `https://trace.wandb.ai`
- **Primary Endpoint**: `/calls/stream_query`
- **Method**: POST
- **Response Format**: JSONL (streaming)

## Authentication

The API uses HTTP Basic Authentication with the W&B API key:

```python
import base64
import os

api_key = os.environ.get("WANDB_API_KEY")
auth_string = base64.b64encode(f"api:{api_key}".encode()).decode()
headers = {
    "Authorization": f"Basic {auth_string}",
    "Content-Type": "application/json"
}
```

## Query Structure

### Basic Query Format

```json
{
    "entity_name": "wandb-applied-ai-team",
    "project_name": "eval-failures",
    "filters": {
        "op": "AndOperation",
        "operands": [
            {"op": "EqOperation", "field": "op_name", "value": "weave.evaluation"},
            {"op": "ContainsOperation", "field": "call_id", "value": "0197a72d-2704-7ced-8c07-0fa1e0ab0557"}
        ]
    },
    "columns": ["id", "started_at", "inputs", "outputs", "summary"],
    "limit": 100
}
```

### Filter Operations

The API supports several filter operations:

1. **EqOperation**: Exact match
   ```json
   {"op": "EqOperation", "field": "call_id", "value": "exact-id"}
   ```

2. **ContainsOperation**: Substring match
   ```json
   {"op": "ContainsOperation", "field": "call_id", "value": "partial-id"}
   ```

3. **AndOperation**: Combine multiple filters with AND logic
   ```json
   {
       "op": "AndOperation",
       "operands": [filter1, filter2, ...]
   }
   ```

4. **OrOperation**: Combine multiple filters with OR logic
   ```json
   {
       "op": "OrOperation",
       "operands": [filter1, filter2, ...]
   }
   ```

### Available Columns

Standard columns that can be selected:
- `id`: Unique identifier for the call
- `project_id`: Project identifier
- `trace_id`: Trace identifier
- `parent_id`: Parent call ID (for nested calls)
- `op_name`: Operation name (e.g., "weave.evaluation")
- `display_name`: Human-readable name
- `started_at`: Timestamp when the call started
- `ended_at`: Timestamp when the call ended
- `inputs`: Input parameters to the call
- `outputs`: Output/results from the call
- `exception`: Any exception information
- `attributes`: Additional metadata
- `summary`: Summary statistics and metadata
- `wb_user_id`: User identifier
- `wb_run_id`: Run identifier

## Implementation Architecture

The wandb-mcp-server implementation uses a layered architecture:

1. **Client Layer**: Handles HTTP communication
2. **Service Layer**: Business logic for trace queries
3. **Query Builder**: Constructs query expressions
4. **Models**: Data structures for requests/responses

## Minimal Implementation Requirements

For a minimal implementation to query evaluation traces by call_id:

1. **Authentication**: Handle W&B API key authentication
2. **Query Construction**: Build proper filter expressions for call_id
3. **HTTP Client**: Make POST requests to the stream_query endpoint
4. **Response Parsing**: Parse JSONL responses
5. **Column Selection**: Allow specifying which columns to retrieve

## Example Usage

To query an evaluation with a specific call_id:

```python
import requests
import json

def query_weave_evaluation(call_id, columns=None):
    base_url = "https://trace.wandb.ai"
    endpoint = f"{base_url}/calls/stream_query"
    
    # Build query
    query = {
        "entity_name": "wandb-applied-ai-team",
        "project_name": "eval-failures",
        "filters": {
            "op": "AndOperation",
            "operands": [
                {"op": "EqOperation", "field": "op_name", "value": "weave.evaluation"},
                {"op": "ContainsOperation", "field": "call_id", "value": call_id}
            ]
        },
        "columns": columns or ["id", "started_at", "inputs", "outputs", "summary"],
        "limit": 100
    }
    
    # Make request
    response = requests.post(endpoint, json=query, headers=headers, stream=True)
    
    # Parse JSONL response
    results = []
    for line in response.iter_lines():
        if line:
            results.append(json.loads(line))
    
    return results
```

## Key Insights

1. **Streaming Response**: The API returns data as JSONL to handle large datasets efficiently
2. **Flexible Filtering**: The operation-based filter system allows complex queries
3. **Column Selection**: Specifying columns reduces data transfer and improves performance
4. **Evaluation Queries**: Use `op_name = "weave.evaluation"` to filter for evaluation calls
5. **Call ID Filtering**: Can use exact match or substring match for call IDs
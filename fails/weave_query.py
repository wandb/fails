"""
Minimal implementation for querying Weave evaluation traces.
"""

import os
import json
import base64
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union, Iterator
from enum import Enum

import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class TraceDepth(Enum):
    """Options for controlling trace retrieval depth."""
    EVALUATION_ONLY = "evaluation_only"  # Only the top-level evaluation trace
    DIRECT_CHILDREN = "direct_children"  # Evaluation + direct children only
    ALL_DESCENDANTS = "all_descendants"  # Evaluation + all descendants (children, grandchildren, etc.)


@dataclass
class WeaveQueryConfig:
    """Configuration for Weave queries."""
    entity_name: str
    project_name: str
    api_key: Optional[str] = None
    base_url: str = "https://trace.wandb.ai"
    
    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.environ.get("WANDB_API_KEY")
            if not self.api_key:
                raise ValueError("WANDB_API_KEY not found in environment variables")


class WeaveQueryClient:
    """Minimal client for querying Weave traces."""
    
    def __init__(self, config: WeaveQueryConfig):
        self.config = config
        self._setup_headers()
    
    def _setup_headers(self):
        """Set up authentication headers."""
        # Note the colon prefix in the auth string - this is important!
        auth_string = base64.b64encode(f":{self.config.api_key}".encode()).decode()
        self.headers = {
            "Authorization": f"Basic {auth_string}",
            "Content-Type": "application/json",
            "Accept": "application/jsonl"
        }
    
    def _execute_query(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute a query to the Weave API.
        
        Args:
            query: The query dictionary
            
        Returns:
            List of trace dictionaries from the JSONL response
        """
        endpoint = f"{self.config.base_url}/calls/stream_query"
        
        try:
            # Make request with stream=True (this works in our test)
            response = requests.post(
                endpoint, 
                headers=self.headers,
                data=json.dumps(query),
                timeout=30,
                stream=True
            )
            
            # Check for errors
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                print(f"Response: {response.text}")
                response.raise_for_status()
            
            # Parse JSONL response
            results = []
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        results.append(data)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line: {e}")
                        continue
            
            return results
                    
        except Exception as e:
            print(f"Query error: {type(e).__name__}: {e}")
            raise
    
    def query_by_call_id(
        self, 
        call_id: str, 
        columns: Optional[List[str]] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Query Weave traces by call ID.
        
        Args:
            call_id: The call ID to filter by (can be partial)
            columns: List of columns to retrieve. If None, uses default set.
            limit: Maximum number of results to return
            
        Returns:
            List of trace dictionaries
        """
        if columns is None:
            columns = [
                "id", "trace_id", "parent_id",
                "started_at", "ended_at",
                "op_name", "display_name",
                "inputs",
                "output",
                "summary", "exception",
                "attributes"
            ]
        
        # Build query with filters
        query = {
            "project_id": f"{self.config.entity_name}/{self.config.project_name}",
            "filters": {
                "op": "EqOperation",
                "field": "id",
                "value": call_id
            },
            "columns": columns,
            "limit": limit
        }
        
        return self._execute_query(query)
    
    def query_evaluation_children(
        self,
        evaluation_call_id: str,
        columns: Optional[List[str]] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Query all child calls of an evaluation.
        
        Args:
            evaluation_call_id: The evaluation call ID
            columns: Columns to retrieve
            limit: Maximum number of results
            
        Returns:
            List of child trace dictionaries
        """
        if columns is None:
            columns = [
                "id", "trace_id", "parent_id",
                "started_at", "ended_at",
                "op_name", "display_name",
                "inputs",
                "output",  # Changed from "outputs" to "output" - the correct column name
                "summary", "exception"
            ]
        
        # Query for calls where parent_id matches the evaluation
        query = {
            "project_id": f"{self.config.entity_name}/{self.config.project_name}",
            "filters": {
                "op": "EqOperation",
                "field": "parent_id",
                "value": evaluation_call_id
            },
            "columns": columns,
            "limit": limit,
            "sort_by": [{"field": "started_at", "direction": "asc"}]
        }
        
        return self._execute_query(query)
    
    def query_descendants_recursive(
        self,
        parent_id: str,
        columns: Optional[List[str]] = None,
        limit_per_level: int = 1000,
        max_depth: Optional[int] = None,
        current_depth: int = 0
    ) -> Dict[str, Any]:
        """
        Recursively query all descendants of a trace.
        
        Args:
            parent_id: The parent trace ID
            columns: Columns to retrieve
            limit_per_level: Maximum results per level
            max_depth: Maximum depth to traverse (None for unlimited)
            current_depth: Current recursion depth
            
        Returns:
            Dictionary with 'traces' list and 'tree' structure
        """
        if max_depth is not None and current_depth >= max_depth:
            return {"traces": [], "tree": {}}
        
        # Get direct children
        children = self.query_evaluation_children(
            evaluation_call_id=parent_id,
            columns=columns,
            limit=limit_per_level
        )
        
        all_traces = []
        tree_structure = {}
        
        for child in children:
            child_id = child["id"]
            all_traces.append(child)
            
            # Recursively get descendants
            descendants = self.query_descendants_recursive(
                parent_id=child_id,
                columns=columns,
                limit_per_level=limit_per_level,
                max_depth=max_depth,
                current_depth=current_depth + 1
            )
            
            all_traces.extend(descendants["traces"])
            tree_structure[child_id] = {
                "trace": child,
                "children": descendants["tree"]
            }
        
        return {"traces": all_traces, "tree": tree_structure}
    
    def extract_column_values(
        self,
        traces: List[Dict[str, Any]],
        column_path: str
    ) -> List[Any]:
        """
        Extract values from a specific column path in traces.
        
        Args:
            traces: List of trace dictionaries
            column_path: Dot-separated path to the column (e.g., "outputs.result")
            
        Returns:
            List of values from the specified column
        """
        values = []
        for trace in traces:
            value = trace
            try:
                for key in column_path.split('.'):
                    value = value.get(key, {})
                if value != {}:
                    values.append(value)
            except (AttributeError, TypeError):
                values.append(None)
        
        return values


def build_trace_hierarchy(
    evaluation: Dict[str, Any],
    all_traces: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Build a hierarchical structure from flat trace list.
    
    Args:
        evaluation: The root evaluation trace
        all_traces: List of all trace dictionaries
        
    Returns:
        Hierarchical structure with parent-child relationships
    """
    # Create a map of trace_id to trace for quick lookup
    trace_map = {trace["id"]: trace for trace in all_traces}
    trace_map[evaluation["id"]] = evaluation
    
    # Build the hierarchy
    hierarchy = {
        "trace": evaluation,
        "children": []
    }
    
    # Group traces by parent_id
    children_by_parent = {}
    for trace in all_traces:
        parent_id = trace.get("parent_id")
        if parent_id:
            if parent_id not in children_by_parent:
                children_by_parent[parent_id] = []
            children_by_parent[parent_id].append(trace)
    
    def build_subtree(node_id: str) -> List[Dict[str, Any]]:
        """Recursively build subtree for a node."""
        children = []
        if node_id in children_by_parent:
            for child_trace in children_by_parent[node_id]:
                child_node = {
                    "trace": child_trace,
                    "children": build_subtree(child_trace["id"])
                }
                children.append(child_node)
        return children
    
    hierarchy["children"] = build_subtree(evaluation["id"])
    return hierarchy


def query_evaluation_data(
    eval_id: str,
    entity_name: str = "wandb-applied-ai-team", 
    project_name: str = "eval-failures",
    columns: Optional[List[str]] = None,
    include_children: bool = True,
    include_outputs: bool = True,
    trace_depth: TraceDepth = TraceDepth.DIRECT_CHILDREN,
    include_hierarchy: bool = True,
    limit: int | None = 1000
) -> Dict[str, Any]:
    """
    Query Weave evaluation data by evaluation ID.
    
    Args:
        eval_id: The evaluation call ID to query
        entity_name: W&B entity name (default: wandb-applied-ai-team)
        project_name: W&B project name (default: eval-failures)
        columns: List of columns to retrieve. If None, uses a default set.
        include_children: Whether to also query child traces (legacy parameter, use trace_depth instead)
        include_outputs: Whether to include output column
        trace_depth: Control the depth of trace retrieval (EVALUATION_ONLY, DIRECT_CHILDREN, or ALL_DESCENDANTS)
        include_hierarchy: Whether to include hierarchical structure in response
        limit: Maximum number of traces to retrieve per level. None means no limit. (default: 1000)
        
    Returns:
        Dictionary containing:
        - evaluation: The evaluation trace data (Dict[str, Any])
        - children: List of child traces (List[Dict[str, Any]]) if trace_depth != EVALUATION_ONLY
        - all_descendants: List of all descendant traces if trace_depth == ALL_DESCENDANTS
        - hierarchy: Hierarchical structure showing parent-child relationships (if include_hierarchy=True)
        - trace_count: Dictionary with counts by level
    """
    # Configuration
    config = WeaveQueryConfig(
        entity_name=entity_name,
        project_name=project_name
    )
    
    # Create client
    client = WeaveQueryClient(config)
    
    # Default columns if not specified
    if columns is None:
        columns = ["id", "op_name", "display_name", "started_at", "ended_at", "inputs", "summary", "parent_id", "trace_id"]
        if include_outputs:
            columns.append("output")  # Changed from "outputs" to "output"
    
    # Query by evaluation ID - use limit for the initial query too
    query_kwargs = {
        "call_id": eval_id,
        "columns": columns
    }
    if limit is not None:
        query_kwargs["limit"] = limit
    
    evaluation_traces = client.query_by_call_id(**query_kwargs)
    
    if not evaluation_traces:
        # If no exact match, try querying by trace_id to find all traces in this evaluation
        query_kwargs["field"] = "trace_id"
        trace_results = client._execute_query({
            "project_id": f"{entity_name}/{project_name}",
            "filters": {
                "op": "EqOperation",
                "field": "trace_id",
                "value": eval_id
            },
            "columns": columns,
            "limit": limit if limit is not None else 1000,
            "sort_by": [{"field": "started_at", "direction": "asc"}]
        })
        
        if trace_results:
            evaluation_traces = trace_results
        else:
            raise ValueError(f"No evaluation found with ID: {eval_id}")
    
    # Find the actual evaluation trace (root trace with parent_id=None)
    eval_trace = None
    
    # First look for a root trace (parent_id is None) with Evaluation.evaluate
    for trace in evaluation_traces:
        if trace.get('parent_id') is None and "Evaluation.evaluate" in trace.get('op_name', ''):
            eval_trace = trace
            break
    
    # If not found, look for any root trace (parent_id is None)
    if not eval_trace:
        for trace in evaluation_traces:
            if trace.get('parent_id') is None:
                eval_trace = trace
                break
    
    # If still not found, the eval_id might be pointing to a child trace
    # In this case, we need to find the root by following parent_id chain
    if not eval_trace and evaluation_traces:
        # Take the first trace and follow its parent chain
        current_trace = evaluation_traces[0]
        
        # If this trace has no parent, it's the root
        if current_trace.get('parent_id') is None:
            eval_trace = current_trace
        else:
            # Follow the parent chain to find the root
            parent_id = current_trace.get('parent_id')
            while parent_id:
                parent_results = client.query_by_call_id(parent_id, columns=columns, limit=1)
                if parent_results:
                    parent_trace = parent_results[0]
                    if parent_trace.get('parent_id') is None:
                        # Found the root
                        eval_trace = parent_trace
                        break
                    else:
                        # Keep going up
                        parent_id = parent_trace.get('parent_id')
                else:
                    # Can't find parent, use what we have
                    break
    
    if not eval_trace:
        # Last resort - use the first trace we found
        eval_trace = evaluation_traces[0]
        print(f"Warning: Could not find root evaluation trace. Using trace with parent_id={eval_trace.get('parent_id')}")
    
    result: Dict[str, Any] = {"evaluation": eval_trace}
    
    # Handle different trace depth options
    if trace_depth == TraceDepth.EVALUATION_ONLY:
        # Only return the evaluation trace
        result["trace_count"] = {"evaluation": 1, "total": 1}
        
    elif trace_depth == TraceDepth.DIRECT_CHILDREN:
        # Query direct children only
        try:
            child_query_kwargs = {
                "evaluation_call_id": eval_trace['id'],
                "columns": columns
            }
            if limit is not None:
                child_query_kwargs["limit"] = limit
                
            child_traces = client.query_evaluation_children(**child_query_kwargs)
            
            # Filter out any traces that aren't actual children
            # (sometimes the API returns traces with parent_id=None or wrong parent_id)
            actual_children = [
                trace for trace in child_traces 
                if trace.get('parent_id') == eval_trace['id']
            ]
            
            result["children"] = actual_children
            result["trace_count"] = {
                "evaluation": 1,
                "direct_children": len(actual_children),
                "total": 1 + len(actual_children)
            }
            
            if include_hierarchy:
                result["hierarchy"] = build_trace_hierarchy(eval_trace, actual_children)
                
        except Exception as e:
            print(f"Warning: Could not query child traces: {e}")
            result["children"] = []
            result["trace_count"] = {"evaluation": 1, "total": 1}
            
    elif trace_depth == TraceDepth.ALL_DESCENDANTS:
        # Recursively query all descendants
        try:
            descendants_kwargs = {
                "parent_id": eval_trace['id'],
                "columns": columns
            }
            if limit is not None:
                descendants_kwargs["limit_per_level"] = limit
                
            descendants_data = client.query_descendants_recursive(**descendants_kwargs)
            
            all_traces = descendants_data["traces"]
            
            # Filter to ensure we only include actual descendants
            # (exclude any traces that don't have a proper parent chain)
            actual_descendants = []
            for trace in all_traces:
                # Check if this trace is actually a descendant by verifying parent_id is not None
                if trace.get('parent_id') is not None:
                    actual_descendants.append(trace)
            
            # Separate direct children from all descendants
            direct_children = [
                t for t in actual_descendants 
                if t.get("parent_id") == eval_trace['id']
            ]
            
            result["children"] = direct_children
            result["all_descendants"] = actual_descendants
            
            # Count traces by level
            trace_count = {"evaluation": 1, "direct_children": len(direct_children)}
            level_counts = {}
            
            for trace in actual_descendants:
                # Count depth by following parent chain
                depth = 1
                current_parent = trace.get("parent_id")
                while current_parent and current_parent != eval_trace['id']:
                    depth += 1
                    # Find parent in all_traces
                    parent_trace = next((t for t in actual_descendants if t["id"] == current_parent), None)
                    if parent_trace:
                        current_parent = parent_trace.get("parent_id")
                    else:
                        break
                
                level_key = f"level_{depth}_descendants"
                level_counts[level_key] = level_counts.get(level_key, 0) + 1
            
            trace_count.update(level_counts)
            trace_count["total"] = 1 + len(actual_descendants)
            result["trace_count"] = trace_count
            
            if include_hierarchy:
                result["hierarchy"] = build_trace_hierarchy(eval_trace, actual_descendants)
                
        except Exception as e:
            print(f"Warning: Could not query descendants: {e}")
            result["children"] = []
            result["all_descendants"] = []
            result["trace_count"] = {"evaluation": 1, "total": 1}
    
    # Legacy support: if include_children is False, override to EVALUATION_ONLY
    if not include_children and trace_depth != TraceDepth.EVALUATION_ONLY:
        print("Warning: include_children=False is deprecated. Use trace_depth=TraceDepth.EVALUATION_ONLY instead.")
        # Remove children data if it was added
        result.pop("children", None)
        result.pop("all_descendants", None)
        result["trace_count"] = {"evaluation": 1, "total": 1}
    
    return result
"""
Minimal implementation for querying Weave evaluation traces.
"""

import os
import json
import base64
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union, Tuple, Type
from enum import Enum
from functools import wraps
import weave
import requests
from requests.exceptions import ConnectionError, Timeout, HTTPError
from dotenv import load_dotenv
from rich.console import Console

from fails.utils import filter_trace_data_by_columns

# Load environment variables from .env file
load_dotenv()

console = Console()


def retry_on_failure(
    max_retries: int = 3,
    backoff_factor: float = 1.0,
    exceptions: Tuple[Type[Exception], ...] = (ConnectionError, Timeout, HTTPError),
    on_retry: Optional[Any] = None
):
    """
    Decorator to retry a function on failure with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        backoff_factor: Factor for exponential backoff (delay = backoff_factor * 2^retry_count)
        exceptions: Tuple of exceptions to catch and retry
        on_retry: Optional callback function called on each retry with (attempt, exception)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    # Don't retry on last attempt
                    if attempt == max_retries:
                        break
                    
                    # Don't retry on 4xx errors (client errors) except 429 (rate limit)
                    if isinstance(e, HTTPError) and e.response is not None:
                        if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                            raise
                    
                    # Calculate delay with exponential backoff
                    delay = backoff_factor * (2 ** attempt)
                    
                    # Call retry callback if provided
                    if on_retry:
                        on_retry(attempt + 1, e)
                    else:
                        # Default logging
                        console.print(f"[yellow]Retry attempt {attempt + 1}/{max_retries} after {type(e).__name__}. Waiting {delay}s...[/yellow]")
                    
                    # Sleep before retrying
                    time.sleep(delay)
            
            # If we get here, all retries failed
            raise last_exception
        
        return wrapper
    return decorator


class TraceDepth(Enum):
    """Options for controlling trace retrieval depth."""

    EVALUATION_ONLY = "evaluation_only"  # Only the top-level evaluation trace
    DIRECT_CHILDREN = "direct_children"  # Evaluation + direct children only
    ALL_DESCENDANTS = "all_descendants"  # Evaluation + all descendants (children, grandchildren, etc.)


@dataclass
class WeaveQueryConfig:
    """Configuration for Weave queries."""

    wandb_entity: str
    wandb_project: str
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
            "Accept": "application/jsonl",
        }

    @weave.op
    def _make_api_request(
        self, 
        url: str, 
        headers: Dict[str, str], 
        data: Union[str, Dict[str, Any]], 
        timeout: int = 30, 
        stream: bool = False
    ) -> requests.Response:
        """
        Make an API request with all parameters explicitly passed.
        
        This method contains all request parameters in its signature to enable
        proper logging via decorators.
        
        Args:
            url: The full URL endpoint
            headers: Request headers
            data: Request payload (either JSON string or dict)
            timeout: Request timeout in seconds
            stream: Whether to stream the response
            
        Returns:
            The requests Response object
            
        Raises:
            requests.HTTPError: If the request fails
        """
        # Convert dict to JSON string if needed
        if isinstance(data, dict):
            data = json.dumps(data)
        
        # Define the actual request function with retry logic
        @retry_on_failure(
            max_retries=3,
            backoff_factor=1.0,
            exceptions=(ConnectionError, Timeout, HTTPError)
        )
        def _make_request():
            response = requests.post(
                url,
                headers=headers,
                data=data,
                timeout=timeout,
                stream=stream
            )
            
            # Check for errors
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                print(f"Response: {response.text}")
                response.raise_for_status()
                
            return response
        
        # Execute the request with retry logic
        return _make_request()

    @weave.op
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
            # Use the common API request method
            response = self._make_api_request(
                url=endpoint,
                headers=self.headers,
                data=query,
                timeout=30,
                stream=True
            )

            # Parse JSONL response
            results = []
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        results.append(data)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line: {e}")
                        continue

            return results

        except Exception as e:
            print(f"Query error: {type(e).__name__}: {e}")
            raise

    def query_by_call_id(
        self, call_id: str, columns: Optional[List[str]] = None, limit: int | None = None
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
                "id",
                "trace_id",
                "parent_id",
                "started_at",
                "ended_at",
                "op_name",
                "display_name",
                "inputs",
                "output",
                "summary",
                "exception",
                "attributes",
            ]

        # Build query with filters
        query = {
            "project_id": f"{self.config.wandb_entity}/{self.config.wandb_project}",
            "filters": {"op": "EqOperation", "field": "id", "value": call_id},
            "columns": columns,
        }

        if limit is not None:
            query["limit"] = limit

        return self._execute_query(query)

    def query_evaluation_children(
        self,
        evaluation_call_id: str,
        columns: Optional[List[str]] = None,
        limit: int | None = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        op_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query all child calls of an evaluation.

        Uses the correct Weave API filter syntax with both filter and query parameters.

        Args:
            evaluation_call_id: The evaluation call ID
            columns: Columns to retrieve
            limit: Maximum number of results
            filter_dict: Optional filter dictionary (e.g., {"output.scores.affiliation_score.correct": False})
            op_name: Optional operation name for filtering

        Returns:
            List of child trace dictionaries
        """
        if columns is None:
            columns = [
                "id",
                "trace_id",
                "parent_id",
                "started_at",
                "ended_at",
                "op_name",
                "display_name",
                "inputs",
                "output",
                "summary",
                "exception",
            ]

        # Build the base filter with parent_ids
        base_filter = {
            "parent_ids": [evaluation_call_id]
        }
        
        # Add op_name to filter if provided
        if op_name:
            base_filter["op_names"] = [op_name]

        # Build the query structure
        query = {
            "project_id": f"{self.config.wandb_entity}/{self.config.wandb_project}",
            "filter": base_filter,
            "columns": columns,
            "sort_by": [{"field": "started_at", "direction": "asc"}],
        }
        
        # If additional filters are provided, use the query parameter with $expr
        if filter_dict:
            # Build $expr query for complex filtering
            expr_conditions = []
            
            for field, value in filter_dict.items():
                # Convert boolean values to string for $literal
                literal_value = str(value).lower() if isinstance(value, bool) else value
                
                expr_conditions.append({
                    "$eq": [
                        {"$getField": field},
                        {"$literal": literal_value}
                    ]
                })
            
            # If multiple conditions, combine with $and
            if len(expr_conditions) == 1:
                query["query"] = {"$expr": expr_conditions[0]}
            else:
                query["query"] = {"$expr": {"$and": expr_conditions}}
            
            # Add sorting for filtered fields
            for field in filter_dict.keys():
                query["sort_by"] = [{"field": field, "direction": "asc"}]
                break  # Just use the first field for sorting

        if limit is not None:
            query["limit"] = limit

        return self._execute_query(query)

    def query_descendants_recursive(
        self,
        parent_id: str,
        columns: Optional[List[str]] = None,
        limit_per_level: int = 1000,
        max_depth: Optional[int] = None,
        current_depth: int = 0,
        filter_dict: Optional[Dict[str, Any]] = None,
        op_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Recursively query all descendants of a trace.

        Args:
            parent_id: The parent trace ID
            columns: Columns to retrieve
            limit_per_level: Maximum results per level
            max_depth: Maximum depth to traverse (None for unlimited)
            current_depth: Current recursion depth
            filter_dict: Optional filter dictionary (e.g., {"column_name": value})

        Returns:
            Dictionary with 'traces' list and 'tree' structure
        """
        if max_depth is not None and current_depth >= max_depth:
            return {"traces": [], "tree": {}}

        # Get direct children
        children = self.query_evaluation_children(
            evaluation_call_id=parent_id, 
            columns=columns, 
            limit=limit_per_level,
            filter_dict=filter_dict,
            op_name=op_name
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
                current_depth=current_depth + 1,
                filter_dict=filter_dict,
                op_name=op_name,
            )

            all_traces.extend(descendants["traces"])
            tree_structure[child_id] = {"trace": child, "children": descendants["tree"]}

        return {"traces": all_traces, "tree": tree_structure}

    def extract_column_values(
        self, traces: List[Dict[str, Any]], column_path: str
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
                for key in column_path.split("."):
                    value = value.get(key, {})
                if value != {}:
                    values.append(value)
            except (AttributeError, TypeError):
                values.append(None)

        return values

    # ----------  Refs utilities  ----------
    @staticmethod
    def _collect_refs(obj: Any) -> List[str]:
        """
        Recursively collect all Weave references (strings starting with 'weave:///').
        
        Args:
            obj: Any Python object to search for refs
            
        Returns:
            List of unique Weave reference strings
        """
        refs = set()
        
        def _collect(item: Any):
            if isinstance(item, str):
                if item.startswith("weave:///"):
                    refs.add(item)
            elif isinstance(item, dict):
                for value in item.values():
                    _collect(value)
            elif isinstance(item, (list, tuple, set)):
                for value in item:
                    _collect(value)
        
        _collect(obj)
        return list(refs)

    @weave.op
    def extract_refs_from_traces(
        self, 
        traces: List[Dict[str, Any]],
        deep: bool = False
    ) -> Dict[str, Dict[str, Union[str, List[str]]]]:
        """
        Extract Weave references from traces.
        
        Args:
            traces: List of trace dictionaries
            deep: If True, extracts all refs recursively. If False, only extracts
                  top-level refs with clean paths.
        
        Returns:
            Dictionary mapping trace_id to refs found:
            - If deep=False: {trace_id: {path: ref_or_list_of_refs}}
            - If deep=True: {trace_id: {container: [all_refs_in_container]}}
        """
        trace_refs: Dict[str, Dict[str, Union[str, List[str]]]] = {}
        
        for trace in traces:
            refs_here: Dict[str, Union[str, List[str]]] = {}
            containers = {
                "inputs": trace.get("inputs", {}),
                "output": trace.get("output", {}),
                "attributes": trace.get("attributes", {}),
            }
            
            if deep:
                # Deep extraction - collect all refs in each container
                for container_name, container in containers.items():
                    found_refs = self._collect_refs(container)
                    if found_refs:
                        refs_here[container_name] = found_refs
            else:
                # Shallow extraction - only top-level with clean paths
                for container_name, container in containers.items():
                    if isinstance(container, dict):
                        for key, value in container.items():
                            path = f"{container_name}.{key}"
                            
                            # Handle direct string refs
                            if isinstance(value, str) and value.startswith("weave:///"):
                                refs_here[path] = value
                            
                            # Handle lists of refs (common pattern)
                            elif isinstance(value, list):
                                ref_list = [
                                    item for item in value 
                                    if isinstance(item, str) and item.startswith("weave:///")
                                ]
                                if ref_list:
                                    refs_here[path] = ref_list if len(ref_list) > 1 else ref_list[0]
                            
                            # Handle dict with ref values (e.g., {"model": "weave:///..."})
                            elif isinstance(value, dict):
                                nested_refs = {}
                                for nested_key, nested_val in value.items():
                                    if isinstance(nested_val, str) and nested_val.startswith("weave:///"):
                                        nested_refs[f"{path}.{nested_key}"] = nested_val
                                refs_here.update(nested_refs)
            
            if refs_here:
                trace_refs[trace["id"]] = refs_here
        
        return trace_refs

    def read_refs_batch(self, refs: List[str]) -> List[Any]:
        """
        Resolve Weave references in batch via the /refs/read_batch endpoint.
        
        Args:
            refs: List of Weave reference strings to resolve
            
        Returns:
            List of resolved values in the same order as input refs
            
        Raises:
            requests.HTTPError: If the API request fails
        """
        if not refs:
            return []
        
        # Deduplicate while preserving order
        seen = set()
        unique_refs = []
        for ref in refs:
            if ref not in seen:
                seen.add(ref)
                unique_refs.append(ref)
        
        endpoint = f"{self.config.base_url}/refs/read_batch"
        payload = {"refs": unique_refs}
        
        # Note: refs endpoint uses regular JSON, not JSONL
        headers = {
            "Authorization": self.headers["Authorization"],
            "Content-Type": "application/json",
            "Accept": "application/json",  # Not JSONL for refs endpoint
        }
        
        try:
            # Use the common API request method
            response = self._make_api_request(
                url=endpoint,
                headers=headers,
                data=payload,
                timeout=30,
                stream=False
            )
            
            # Get values in order of unique_refs
            result_values = response.json()["vals"]
            
            # Map back to original refs order (handling duplicates)
            ref_to_value = dict(zip(unique_refs, result_values))
            return [ref_to_value[ref] for ref in refs]
            
        except requests.HTTPError as e:
            print(f"Error resolving refs: {e}")
            print(f"Response: {e.response.text if e.response else 'No response'}")
            raise
        except Exception as e:
            print(f"Unexpected error resolving refs: {type(e).__name__}: {e}")
            raise

    @staticmethod
    def replace_refs_in_object(obj: Any, resolved_refs: Dict[str, Any]) -> Any:
        """
        Recursively replace Weave refs in an object with their resolved values.
        
        Args:
            obj: Any Python object that might contain refs
            resolved_refs: Dict mapping ref strings to resolved values
            
        Returns:
            A new object with refs replaced by their resolved values
        """
        import copy
        
        if isinstance(obj, str):
            # If it's a ref string and we have a resolved value, return the resolved value
            if obj.startswith("weave:///") and obj in resolved_refs:
                return copy.deepcopy(resolved_refs[obj])
            return obj
            
        elif isinstance(obj, dict):
            # Recursively process dictionary
            result = {}
            for key, value in obj.items():
                result[key] = WeaveQueryClient.replace_refs_in_object(value, resolved_refs)
            return result
            
        elif isinstance(obj, list):
            # Recursively process list
            return [WeaveQueryClient.replace_refs_in_object(item, resolved_refs) for item in obj]
            
        elif isinstance(obj, tuple):
            # Recursively process tuple (return as list since tuples are immutable)
            return tuple(WeaveQueryClient.replace_refs_in_object(item, resolved_refs) for item in obj)
            
        elif isinstance(obj, set):
            # Recursively process set
            return {WeaveQueryClient.replace_refs_in_object(item, resolved_refs) for item in obj}
            
        else:
            # For other types (int, float, bool, None, etc.), return as-is
            return obj

    def replace_refs_in_traces(
        self, 
        traces: List[Dict[str, Any]], 
        resolved_refs: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Replace all Weave refs in traces with their resolved values.
        
        Args:
            traces: List of trace dictionaries
            resolved_refs: Dict mapping ref strings to resolved values
            
        Returns:
            New list of traces with refs replaced
        """
        import copy
        
        # Create deep copies of traces to avoid modifying originals
        updated_traces = []
        
        for trace in traces:
            # Create a copy of the trace
            updated_trace = copy.deepcopy(trace)
            
            # Replace refs in key containers
            for container in ["inputs", "output", "attributes"]:
                if container in updated_trace and updated_trace[container]:
                    updated_trace[container] = self.replace_refs_in_object(
                        updated_trace[container], resolved_refs
                    )
            
            updated_traces.append(updated_trace)
        
        return updated_traces

@weave.op
def build_trace_hierarchy(
    evaluation: Dict[str, Any], all_traces: List[Dict[str, Any]]
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
    hierarchy = {"trace": evaluation, "children": []}

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
                    "children": build_subtree(child_trace["id"]),
                }
                children.append(child_node)
        return children

    hierarchy["children"] = build_subtree(evaluation["id"])
    return hierarchy

@weave.op
def query_evaluation_trace_data(
    eval_id: str,
    wandb_entity: str,
    wandb_project: str,
    columns: Optional[List[str]] = None,
    include_outputs: bool = True,
    trace_depth: Union[TraceDepth, bool, None] = TraceDepth.DIRECT_CHILDREN,
    include_hierarchy: bool = True,
    limit: int | None = None,
    filter_dict: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Query Weave evaluation data by evaluation ID.

    Args:
        trace_depth: Controls trace retrieval depth. Can be:
            - TraceDepth enum value (EVALUATION_ONLY, DIRECT_CHILDREN, ALL_DESCENDANTS)
            - bool: False maps to EVALUATION_ONLY, True maps to DIRECT_CHILDREN
            - None: defaults to DIRECT_CHILDREN
        include_children: DEPRECATED - use trace_depth instead
        filter_dict: Optional filter dictionary for filtering child traces (e.g., {"output.scores.correct": False})
    """

    # Handle flexible trace_depth input
    if isinstance(trace_depth, bool):
        # Boolean backward compatibility
        trace_depth = (
            TraceDepth.EVALUATION_ONLY
            if not trace_depth
            else TraceDepth.DIRECT_CHILDREN
        )
    elif trace_depth is None:
        # None defaults to DIRECT_CHILDREN
        trace_depth = TraceDepth.DIRECT_CHILDREN
    elif not isinstance(trace_depth, TraceDepth):
        raise ValueError(
            f"trace_depth must be TraceDepth enum, bool, or None. Got: {type(trace_depth)}"
        )

    # Configuration
    config = WeaveQueryConfig(wandb_entity=wandb_entity, wandb_project=wandb_project)

    # Create client
    client = WeaveQueryClient(config)

    # Default columns if not specified
    if columns is None:
        columns = [
            "id",
            "parent_id",
            "trace_id",
            "op_name",
            "display_name",
            "started_at",
            "ended_at",
            "inputs",
            "summary",
            "exception",
        ]
        if include_outputs:
            columns.append("output")

    # Query by evaluation ID - use limit for the initial query too
    query_kwargs = {"call_id": eval_id, "columns": columns}
    if limit is not None:
        query_kwargs["limit"] = limit

    evaluation_traces = client.query_by_call_id(**query_kwargs)
    if not evaluation_traces:
        console.print(f"[red]No evaluation found with ID: {eval_id}[/red]")
        raise ValueError(f"No evaluation found with ID: {eval_id}")

    # Find the actual evaluation trace (root trace with parent_id=None)
    eval_trace = None

    # First look for a root trace (parent_id is None) with Evaluation.evaluate
    for trace in evaluation_traces:
        if trace.get("parent_id") is None and "Evaluation.evaluate" in trace.get(
            "op_name", ""
        ):
            eval_trace = trace
            break

    # If not found, look for any root trace (parent_id is None)
    if not eval_trace:
        for trace in evaluation_traces:
            if trace.get("parent_id") is None:
                eval_trace = trace
                break

    # If still not found, the eval_id might be pointing to a child trace
    # In this case, we need to find the root by following parent_id chain
    if not eval_trace and evaluation_traces:
        # Take the first trace and follow its parent chain
        current_trace = evaluation_traces[0]

        # If this trace has no parent, it's the root
        if current_trace.get("parent_id") is None:
            eval_trace = current_trace
        else:
            # Follow the parent chain to find the root
            parent_id = current_trace.get("parent_id")
            while parent_id:
                parent_results = client.query_by_call_id(
                    parent_id, columns=columns, limit=1
                )
                if parent_results:
                    parent_trace = parent_results[0]
                    if parent_trace.get("parent_id") is None:
                        # Found the root
                        eval_trace = parent_trace
                        break
                    else:
                        # Keep going up
                        parent_id = parent_trace.get("parent_id")
                else:
                    # Can't find parent, use what we have
                    break

    if not eval_trace:
        # Last resort - use the first trace we found
        eval_trace = evaluation_traces[0]
        print(
            f"Warning: Could not find root evaluation trace. Using trace with parent_id={eval_trace.get('parent_id')}"
        )

    result: Dict[str, Any] = {"evaluation": eval_trace}

    # Handle different trace depth options
    if trace_depth == TraceDepth.EVALUATION_ONLY:
        # Only return the evaluation trace
        result["trace_count"] = {"evaluation": 1, "total": 1}

    elif trace_depth == TraceDepth.DIRECT_CHILDREN:
        # Query direct children only
        try:
            # If we have a filter_dict, we need to get the op_name first
            op_name = None
            if filter_dict:
                # Get one child to find the op_name
                sample_children = client.query_evaluation_children(
                    evaluation_call_id=eval_trace["id"],
                    columns=["op_name"],
                    limit=1
                )
                if sample_children:
                    op_name = sample_children[0].get("op_name")
            
            child_query_kwargs = {
                "evaluation_call_id": eval_trace["id"],
                "columns": columns,
            }
            if limit is not None:
                child_query_kwargs["limit"] = limit
            if filter_dict is not None:
                child_query_kwargs["filter_dict"] = filter_dict
            if op_name is not None:
                child_query_kwargs["op_name"] = op_name

            child_traces = client.query_evaluation_children(**child_query_kwargs)

            result["children"] = child_traces
            result["trace_count"] = {
                "evaluation": 1,
                "direct_children": len(child_traces),
                "total": 1 + len(child_traces),
            }

            if include_hierarchy:
                result["hierarchy"] = build_trace_hierarchy(eval_trace, child_traces)

        except Exception as e:
            print(f"Warning: Could not query child traces: {e}")
            result["children"] = []
            result["trace_count"] = {"evaluation": 1, "total": 1}

    elif trace_depth == TraceDepth.ALL_DESCENDANTS:
        # Recursively query all descendants
        try:
            # If we have a filter_dict, we need to get the op_name first
            op_name = None
            if filter_dict:
                # Get one child to find the op_name
                sample_children = client.query_evaluation_children(
                    evaluation_call_id=eval_trace["id"],
                    columns=["op_name"],
                    limit=1
                )
                if sample_children:
                    op_name = sample_children[0].get("op_name")
            
            descendants_kwargs = {"parent_id": eval_trace["id"], "columns": columns}
            if limit is not None:
                descendants_kwargs["limit_per_level"] = limit
            if filter_dict is not None:
                descendants_kwargs["filter_dict"] = filter_dict
            if op_name is not None:
                descendants_kwargs["op_name"] = op_name

            descendants_data = client.query_descendants_recursive(**descendants_kwargs)

            all_traces = descendants_data["traces"]

            # Filter to ensure we only include actual descendants
            # (exclude any traces that don't have a proper parent chain)
            actual_descendants = []
            for trace in all_traces:
                # Check if this trace is actually a descendant by verifying parent_id is not None
                if trace.get("parent_id") is not None:
                    actual_descendants.append(trace)

            # Separate direct children from all descendants
            direct_children = [
                t for t in actual_descendants if t.get("parent_id") == eval_trace["id"]
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
                while current_parent and current_parent != eval_trace["id"]:
                    depth += 1
                    # Find parent in all_traces
                    parent_trace = next(
                        (t for t in actual_descendants if t["id"] == current_parent),
                        None,
                    )
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
                result["hierarchy"] = build_trace_hierarchy(
                    eval_trace, actual_descendants
                )

        except Exception as e:
            print(f"Warning: Could not query descendants: {e}")
            result["children"] = []
            result["all_descendants"] = []
            result["trace_count"] = {"evaluation": 1, "total": 1}

    return result

@weave.op
def query_evaluation_data(
    eval_id: str,
    wandb_entity: str,
    wandb_project: str,
    columns: Optional[List[str]] = None,
    include_outputs: bool = True,
    trace_depth: Union[TraceDepth, bool, None] = TraceDepth.DIRECT_CHILDREN,
    include_hierarchy: bool = True,
    limit: int | None = None,
    resolve_refs: bool = True,
    replace_refs: bool = True,
    deep_ref_extraction: bool = False,
    filter_dict: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Query Weave evaluation data with optional reference resolution.
    
    This is a convenience wrapper around query_evaluation_trace_data that also
    extracts and resolves Weave references found in the traces.
    
    Args:
        Same as query_evaluation_data, plus:
        resolve_refs: If True, automatically resolve all Weave refs found
        replace_refs: If True, replace refs in traces with their resolved values
        deep_ref_extraction: If True, extract all nested refs. If False,
                           only extract top-level refs with clean paths.
        filter_dict: Optional filter dictionary for filtering child traces (e.g., {"output.scores.correct": False})
    
    Returns:
        Same as query_evaluation_data, plus:
        - refs_by_trace: Mapping of trace_id to refs found
        - resolved_refs: Dict mapping ref strings to resolved values (if resolve_refs=True)
        Note: If replace_refs=True, all ref strings in traces are replaced with
              their resolved values
    """

    if wandb_entity == "" or wandb_project == "":
        raise ValueError("wandb_entity and wandb_project must be set")
    
    # Ensure we include the necessary columns for ref extraction
    if columns is None:
        columns = [
            "id",
            "parent_id", 
            "trace_id",
            "op_name",
            "display_name",
            "started_at",
            "ended_at",
            "inputs",
            "attributes",  # Important for refs
            "summary",
            "exception",
        ]
        if include_outputs:
            columns.append("output")
    else:
        # Ensure we have the columns needed for ref extraction
        for col in ["inputs", "output", "attributes"]:
            if col not in columns and (col != "output" or include_outputs):
                columns = columns.copy()
                columns.append(col)
    
    # Get base evaluation data
    result = query_evaluation_trace_data(
        eval_id=eval_id,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        columns=columns,
        include_outputs=include_outputs,
        trace_depth=trace_depth,
        include_hierarchy=include_hierarchy,
        limit=limit,
        filter_dict=filter_dict,
    )
    
    # Extract refs from all traces
    all_traces = [result["evaluation"]]
    if "children" in result:
        all_traces.extend(result["children"])
    if "all_descendants" in result:
        # Don't double-count children
        all_traces = [result["evaluation"]] + result["all_descendants"]
    
    # Get the client instance we used
    config = WeaveQueryConfig(wandb_entity=wandb_entity, wandb_project=wandb_project)
    client = WeaveQueryClient(config)
    
    # Extract refs
    refs_by_trace = client.extract_refs_from_traces(all_traces, deep=deep_ref_extraction)
    result["refs_by_trace"] = refs_by_trace
    
    if resolve_refs and refs_by_trace:
        # Collect all unique refs
        all_refs = set()
        for trace_refs in refs_by_trace.values():
            for path, ref_or_refs in trace_refs.items():
                if isinstance(ref_or_refs, list):
                    all_refs.update(ref_or_refs)
                else:
                    all_refs.add(ref_or_refs)
        
        # Resolve in batch
        unique_refs = sorted(all_refs)
        if unique_refs:
            try:
                resolved_values = client.read_refs_batch(unique_refs)
                resolved_refs = dict(zip(unique_refs, resolved_values))
                result["resolved_refs"] = resolved_refs
                
                # Replace refs in all traces with their resolved values if requested
                if replace_refs:
                    updated_traces = client.replace_refs_in_traces(all_traces, resolved_refs)
                    
                    # Find the evaluation trace more robustly
                    eval_trace = None
                    original_eval_id = result["evaluation"]["id"]
                    
                    # First try to find by ID (most reliable)
                    for trace in updated_traces:
                        if trace["id"] == original_eval_id:
                            eval_trace = trace
                            break
                    
                    # If not found by ID, look for evaluation characteristics
                    if not eval_trace:
                        for trace in updated_traces:
                            # Check if it's a root trace with Evaluation.evaluate in op_name
                            if (trace.get("parent_id") is None and 
                                "Evaluation.evaluate" in trace.get("op_name", "")):
                                eval_trace = trace
                                break
                    
                    # If still not found, look for any root trace
                    if not eval_trace:
                        for trace in updated_traces:
                            if trace.get("parent_id") is None:
                                eval_trace = trace
                                break
                    
                    # Last resort - use the first trace
                    if not eval_trace:
                        print("Warning: Could not identify evaluation trace after ref replacement, using first trace")
                        eval_trace = updated_traces[0]
                    
                    # Update the result with the modified traces
                    result["evaluation"] = eval_trace
                    
                    if "children" in result:
                        # Find which traces are direct children
                        child_ids = {t["id"] for t in result["children"]}
                        result["children"] = [t for t in updated_traces if t["id"] in child_ids and t["id"] != eval_trace["id"]]
                    
                    if "all_descendants" in result:
                        # All traces except the evaluation are descendants
                        result["all_descendants"] = [t for t in updated_traces if t["id"] != eval_trace["id"]]
                    
                    # Update hierarchy if present
                    if include_hierarchy and "hierarchy" in result:
                        # Rebuild hierarchy with updated traces
                        if "all_descendants" in result:
                            result["hierarchy"] = build_trace_hierarchy(
                                result["evaluation"], result["all_descendants"]
                            )
                        elif "children" in result:
                            result["hierarchy"] = build_trace_hierarchy(
                                result["evaluation"], result["children"]
                            )
                
            except Exception as e:
                print(f"Warning: Failed to resolve refs: {e}")
                result["resolved_refs"] = {}
    
    return result


@weave.op
def filter_evaluation_data_columns(
    eval_data: Dict[str, Any], 
    selected_columns: List[str]
) -> Dict[str, Any]:
    """
    Filter evaluation data to only include selected column paths.
    
    This function filters the evaluation data structure, including child traces
    and descendants, to only include the specified columns. This is useful for
    reducing the amount of data returned when only specific fields are needed.
    
    Args:
        eval_data: The evaluation data dictionary returned by query_evaluation_data
        selected_columns: List of column paths to keep (e.g., ["inputs.example.call_name"])
        
    Returns:
        The filtered evaluation data dictionary
    """
    
    # Create a copy to avoid modifying the original
    filtered_data = eval_data.copy()
    
    # Filter child traces if present
    if "children" in filtered_data and filtered_data["children"]:
        filtered_data["children"] = filter_trace_data_by_columns(
            filtered_data["children"], 
            selected_columns
        )
    
    # Filter all descendants if present
    if "all_descendants" in filtered_data and filtered_data["all_descendants"]:
        filtered_data["all_descendants"] = filter_trace_data_by_columns(
            filtered_data["all_descendants"],
            selected_columns
        )
    
    # Filter the evaluation trace itself
    if "evaluation" in filtered_data:
        filtered_data["evaluation"] = filter_trace_data_by_columns(
            [filtered_data["evaluation"]], 
            selected_columns
        )[0]
    
    # Rebuild hierarchy if present (it will use the filtered traces)
    if "hierarchy" in filtered_data and "evaluation" in filtered_data:
        if "all_descendants" in filtered_data:
            filtered_data["hierarchy"] = build_trace_hierarchy(
                filtered_data["evaluation"], 
                filtered_data["all_descendants"]
            )
        elif "children" in filtered_data:
            filtered_data["hierarchy"] = build_trace_hierarchy(
                filtered_data["evaluation"], 
                filtered_data["children"]
            )
    
    return filtered_data


def get_available_columns(
    eval_id: str,
    wandb_entity: str,
    wandb_project: str,
    include_nested_paths: bool = True,
    max_nesting_depth: int = 3,
) -> Dict[str, Any]:
    """
    Get all available column names from a sample child trace of an evaluation.
    
    This function queries one child trace with ALL columns to discover what
    fields are available for querying. Useful for presenting column options
    to users.
    
    Args:
        eval_id: The evaluation ID
        wandb_entity: Weave entity name
        wandb_project: Weave project name
        include_nested_paths: If True, also return nested paths like "inputs.model"
        
    Returns:
        Dictionary containing:
        - top_level_columns: List of top-level column names
        - nested_paths: Dict of container to list of nested paths (if include_nested_paths=True)
        - all_columns: Flat list of all column names including nested paths
        - sample_trace: The sample child trace used for discovery
    """
    if wandb_entity == "" or wandb_project == "":
        raise ValueError("wandb_entity and wandb_project must be set")
    
    # Configuration
    config = WeaveQueryConfig(wandb_entity=wandb_entity, wandb_project=wandb_project)
    client = WeaveQueryClient(config)
    
    # First, find the evaluation trace
    eval_traces = client.query_by_call_id(eval_id, columns=["id", "op_name", "parent_id"], limit=2)
    
    if not eval_traces:
        raise ValueError(f"No evaluation found with ID: {eval_id}")
    
    # Find the actual evaluation trace
    eval_trace = None
    for trace in eval_traces:
        if trace.get("parent_id") is None and "Evaluation.evaluate" in trace.get("op_name", ""):
            eval_trace = trace
            break
    
    if not eval_trace:
        # Fall back to any root trace
        for trace in eval_traces:
            if trace.get("parent_id") is None:
                eval_trace = trace
                break
    
    if not eval_trace:
        eval_trace = eval_traces[0]
    
    # Query ONE child trace with NO column filtering to get all available columns
    # We don't specify columns parameter, so the API returns everything
    query = {
        "project_id": f"{wandb_entity}/{wandb_project}",
        "filter": {
            "parent_ids": [eval_trace["id"]]
        },
        "limit": 1,  # Just need one sample
        "sort_by": [{"field": "started_at", "direction": "asc"}],
    }
    
    child_traces = client._execute_query(query)
    
    if not child_traces:
        raise ValueError(f"No child traces found for evaluation {eval_id}")
    
    sample_trace = child_traces[0]
    
    # Extract top-level column names
    top_level_columns = sorted(sample_trace.keys())
    
    # Check for and resolve any Weave references in the trace
    refs_found = client._collect_refs(sample_trace)
    if refs_found:
        try:
            resolved_values = client.read_refs_batch(refs_found)
            resolved_refs = dict(zip(refs_found, resolved_values))
            sample_trace = client.replace_refs_in_object(sample_trace, resolved_refs)
        except Exception as e:
            print(f"Warning: Could not resolve refs in sample trace: {e}")
    
    result = {
        "top_level_columns": top_level_columns,
        "sample_trace": sample_trace,
    }
    
    if include_nested_paths:
        nested_paths = {}
        all_paths = set(top_level_columns)
        
        # Define containers that typically have nested data
        containers = ["inputs", "output", "attributes", "summary"]
        
        for container in containers:
            if container in sample_trace and isinstance(sample_trace[container], dict):
                container_paths = []
                
                def extract_paths(obj: Dict[str, Any], prefix: str = "", depth: int = 0, max_depth: int = 3) -> None:
                    """Recursively extract paths from nested dict."""
                    for key, value in obj.items():
                        # Skip keys that start with underscore
                        if key.startswith('_'):
                            continue
                            
                        path = f"{prefix}.{key}" if prefix else key
                        container_paths.append(path)
                        all_paths.add(f"{container}.{path}")
                        
                        # Go deeper for dicts (up to max_depth levels)
                        if isinstance(value, dict) and depth < max_depth:
                            extract_paths(value, path, depth + 1, max_depth)
                
                extract_paths(sample_trace[container], max_depth=max_nesting_depth)
                
                if container_paths:
                    nested_paths[container] = sorted(container_paths)
        
        result["nested_paths"] = nested_paths
        result["all_columns"] = sorted(all_paths)
    else:
        result["all_columns"] = top_level_columns
    
    # Add some helpful metadata
    result["metadata"] = {
        "eval_id": eval_id,
        "eval_op_name": eval_trace.get("op_name", ""),
        "sample_child_id": sample_trace.get("id", ""),
        "sample_child_op_name": sample_trace.get("op_name", ""),
    }
    
    return result

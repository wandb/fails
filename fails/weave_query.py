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
            "Accept": "application/jsonl",
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
                stream=True,
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
        self, call_id: str, columns: Optional[List[str]] = None, limit: int = 1000
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
            "project_id": f"{self.config.entity_name}/{self.config.project_name}",
            "filters": {"op": "EqOperation", "field": "id", "value": call_id},
            "columns": columns,
            "limit": limit,
        }

        return self._execute_query(query)

    def query_evaluation_children(
        self,
        evaluation_call_id: str,
        columns: Optional[List[str]] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Query all child calls of an evaluation.

        Uses the direct filter approach with parent_ids to ensure only
        direct children are returned.

        Args:
            evaluation_call_id: The evaluation call ID
            columns: Columns to retrieve
            limit: Maximum number of results

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
                "output",  # Changed from "outputs" to "output" - the correct column name
                "summary",
                "exception",
            ]

        # Query for calls where parent_id matches the evaluation
        query = {
            "project_id": f"{self.config.entity_name}/{self.config.project_name}",
            "filter": {
                "parent_ids": [
                    evaluation_call_id
                ]  # Use direct filter instead of query expression
            },
            "columns": columns,
            "limit": limit,
            "sort_by": [{"field": "started_at", "direction": "asc"}],
        }

        return self._execute_query(query)

    def query_descendants_recursive(
        self,
        parent_id: str,
        columns: Optional[List[str]] = None,
        limit_per_level: int = 1000,
        max_depth: Optional[int] = None,
        current_depth: int = 0,
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
            evaluation_call_id=parent_id, columns=columns, limit=limit_per_level
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
            response = requests.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
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


def query_evaluation_trace_data(
    eval_id: str,
    entity_name: str = "wandb-applied-ai-team",
    project_name: str = "eval-failures",
    columns: Optional[List[str]] = None,
    include_outputs: bool = True,
    trace_depth: Union[TraceDepth, bool, None] = TraceDepth.DIRECT_CHILDREN,
    include_hierarchy: bool = True,
    limit: int | None = 1000,
) -> Dict[str, Any]:
    """
    Query Weave evaluation data by evaluation ID.

    Args:
        trace_depth: Controls trace retrieval depth. Can be:
            - TraceDepth enum value (EVALUATION_ONLY, DIRECT_CHILDREN, ALL_DESCENDANTS)
            - bool: False maps to EVALUATION_ONLY, True maps to DIRECT_CHILDREN
            - None: defaults to DIRECT_CHILDREN
        include_children: DEPRECATED - use trace_depth instead
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
    config = WeaveQueryConfig(entity_name=entity_name, project_name=project_name)

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
        # If no exact match, try querying by trace_id to find all traces in this evaluation
        query_kwargs["field"] = "trace_id"
        trace_results = client._execute_query(
            {
                "project_id": f"{entity_name}/{project_name}",
                "filters": {"op": "EqOperation", "field": "trace_id", "value": eval_id},
                "columns": columns,
                "limit": limit
                if limit is not None
                else 10000000,  # basically no limit by default
                "sort_by": [{"field": "started_at", "direction": "asc"}],
            }
        )

        if trace_results:
            evaluation_traces = trace_results
        else:
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
            child_query_kwargs = {
                "evaluation_call_id": eval_trace["id"],
                "columns": columns,
            }
            if limit is not None:
                child_query_kwargs["limit"] = limit

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
            descendants_kwargs = {"parent_id": eval_trace["id"], "columns": columns}
            if limit is not None:
                descendants_kwargs["limit_per_level"] = limit

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


def query_evaluation_data(
    eval_id: str,
    entity_name: str = "",
    project_name: str = "",
    columns: Optional[List[str]] = None,
    include_outputs: bool = True,
    trace_depth: Union[TraceDepth, bool, None] = TraceDepth.DIRECT_CHILDREN,
    include_hierarchy: bool = True,
    limit: int | None = 1000,
    resolve_refs: bool = True,
    replace_refs: bool = True,
    deep_ref_extraction: bool = False,
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
    
    Returns:
        Same as query_evaluation_data, plus:
        - refs_by_trace: Mapping of trace_id to refs found
        - resolved_refs: Dict mapping ref strings to resolved values (if resolve_refs=True)
        Note: If replace_refs=True, all ref strings in traces are replaced with
              their resolved values
    """

    if entity_name == "" or project_name == "":
        raise ValueError("entity_name and project_name must be set")
    
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
        entity_name=entity_name,
        project_name=project_name,
        columns=columns,
        include_outputs=include_outputs,
        trace_depth=trace_depth,
        include_hierarchy=include_hierarchy,
        limit=limit,
    )
    
    # Extract refs from all traces
    all_traces = [result["evaluation"]]
    if "children" in result:
        all_traces.extend(result["children"])
    if "all_descendants" in result:
        # Don't double-count children
        all_traces = [result["evaluation"]] + result["all_descendants"]
    
    # Get the client instance we used
    config = WeaveQueryConfig(entity_name=entity_name, project_name=project_name)
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


def get_available_columns(
    eval_id: str,
    entity_name: str = "",
    project_name: str = "",
    include_nested_paths: bool = True,
) -> Dict[str, Any]:
    """
    Get all available column names from a sample child trace of an evaluation.
    
    This function queries one child trace with ALL columns to discover what
    fields are available for querying. Useful for presenting column options
    to users.
    
    Args:
        eval_id: The evaluation ID
        entity_name: Weave entity name
        project_name: Weave project name
        include_nested_paths: If True, also return nested paths like "inputs.model"
        
    Returns:
        Dictionary containing:
        - top_level_columns: List of top-level column names
        - nested_paths: Dict of container to list of nested paths (if include_nested_paths=True)
        - all_columns: Flat list of all column names including nested paths
        - sample_trace: The sample child trace used for discovery
    """
    if entity_name == "" or project_name == "":
        raise ValueError("entity_name and project_name must be set")
    
    # Configuration
    config = WeaveQueryConfig(entity_name=entity_name, project_name=project_name)
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
        "project_id": f"{entity_name}/{project_name}",
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
                
                def extract_paths(obj: Dict[str, Any], prefix: str = "") -> None:
                    """Recursively extract paths from nested dict."""
                    for key, value in obj.items():
                        path = f"{prefix}.{key}" if prefix else key
                        container_paths.append(path)
                        all_paths.add(f"{container}.{path}")
                        
                        # Go one level deeper for dicts
                        if isinstance(value, dict) and len(path.split('.')) < 3:  # Limit depth
                            extract_paths(value, path)
                
                extract_paths(sample_trace[container])
                
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

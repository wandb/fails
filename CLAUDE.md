# CLAUDE.md - Project Knowledge Base

## Terminal UI Arrow Selector Implementation

### Problem
Creating a working arrow-based selector for terminal UI that doesn't result in garbled/overlapping text. The initial attempts resulted in broken displays where text was overlapping and newlines weren't being handled correctly.

### Root Cause
When using raw terminal mode (`tty.setraw()`), the terminal doesn't automatically handle newlines (`\n`) properly. In raw mode:
- `\n` only moves down one line but doesn't return to the beginning of the line
- Terminal output can overlap if not properly cleared
- Standard print() functions don't work as expected

### Solution
The key insights that made it work:

1. **Use Alternate Screen Buffer**
   ```python
   sys.stdout.write("\033[?1049h")  # Enter alternate screen
   sys.stdout.write("\033[?1049l")  # Exit alternate screen
   ```
   This creates a separate screen (like vim/less do) that doesn't interfere with terminal scrollback.

2. **Clear Lines Before Writing**
   ```python
   sys.stdout.write("\033[K")  # Clear line from cursor to end
   ```
   This prevents text overlap by clearing each line before writing new content.

3. **Use Carriage Return + Line Feed in Raw Mode**
   ```python
   sys.stdout.write("text\r\n")  # NOT just \n
   ```
   In raw mode, we need explicit `\r\n` for proper line endings.

4. **Hide/Show Cursor**
   ```python
   sys.stdout.write("\033[?25l")  # Hide cursor
   sys.stdout.write("\033[?25h")  # Show cursor
   ```
   This prevents cursor flicker during redraws.

5. **Clear Remaining Screen**
   ```python
   sys.stdout.write("\033[J")  # Clear from cursor to end of screen
   ```
   This removes any leftover content below our UI.

### Working Implementation Pattern
```python
# Save terminal state
fd = sys.stdin.fileno()
old_settings = termios.tcgetattr(fd)

try:
    # Setup
    sys.stdout.write("\033[?1049h\033[?25l")  # Alt screen + hide cursor
    tty.setraw(fd)
    
    while True:
        # Clear and redraw
        sys.stdout.write("\033[H")  # Move to home
        sys.stdout.write("\033[Kline 1\r\n")
        sys.stdout.write("\033[Kline 2\r\n")
        sys.stdout.write("\033[J")  # Clear rest
        sys.stdout.flush()
        
        # Handle input
        key = sys.stdin.read(1)
        if key == '\x1b':  # Handle arrow keys
            seq = sys.stdin.read(2)
            # Process escape sequences
            
finally:
    # Restore
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    sys.stdout.write("\033[?25h\033[?1049l")
    sys.stdout.flush()
```

### Key Escape Sequences Used
- `\033[H` - Move cursor to home (0,0)
- `\033[K` - Clear line from cursor to end
- `\033[J` - Clear screen from cursor to end
- `\033[?1049h` - Enter alternate screen buffer
- `\033[?1049l` - Exit alternate screen buffer
- `\033[?25l` - Hide cursor
- `\033[?25h` - Show cursor
- `\033[A` - Up arrow
- `\033[B` - Down arrow

### Lessons Learned
1. Don't use print() in raw mode - use sys.stdout.write()
2. Always use \r\n for line endings in raw mode
3. Clear lines before writing to prevent overlap
4. Use alternate screen buffer for full-screen TUIs
5. Always restore terminal settings in a finally block
6. Flush output after writes for immediate display

This approach creates a clean, flicker-free terminal UI that works across different terminal emulators.

## Additional Terminal UI Best Practices

### Initial Screen Clear
When entering alternate screen buffer, always clear it before starting the main loop:
```python
# Enter alternate screen buffer and hide cursor
sys.stdout.write("\033[?1049h\033[?25l")
sys.stdout.flush()

# Set terminal to raw mode
tty.setraw(sys.stdin.fileno())

# Initial clear of the alternate screen
sys.stdout.write("\033[2J\033[H")
sys.stdout.flush()
```
This prevents any initial garbage or previous content from appearing.

### Terminal Height Considerations
Be aware that users may have terminals of different heights. Design your UI to work well even in shorter terminals:
- Use scrolling windows for long lists
- Show indicators when content extends beyond view
- Consider a reasonable default window size (e.g., 20 items)

### Visual Design Principles
1. **Use Unicode symbols sparingly but effectively**
   - `✓` for selected items is universally understood
   - Play button symbols (▶/▷) work well for cursors
   - Avoid overwhelming users with too many symbols

2. **Color with purpose**
   - Use brand colors when possible (map to nearest ANSI equivalents)
   - Selected items should stand out (bright colors)
   - Unselected items should recede (gray/dim colors)
   - Groups/headers benefit from bold text without color

3. **Clear visual hierarchy**
   - Headers separated by divider lines
   - Empty lines between groups for breathing room
   - Consistent indentation for sub-items

### Common ANSI Color Codes
- `\033[90m` - Dark gray (good for unselected)
- `\033[37m` - Light gray (better readability)
- `\033[96m` - Bright cyan (good for highlights)
- `\033[1m` - Bold text (good for headers)
- `\033[0m` - Reset all formatting

### Handling Long Content
When header text gets long (like navigation instructions), consider:
- Using right-aligned status indicators
- Breaking instructions into multiple lines
- Using separators to create clear zones
- Adjusting separator line lengths to match content

Example layout:
```
==========================================================================================
Column Selection                                                    Selected: 4/21
------------------------------------------------------------------------------------------
↑/↓ Navigate  |  Space Select  |  a All  |  n None  |  q Save
──────────────────────────────────────────────────────────────────────────────────────────
```

This creates distinct zones for title, status, instructions, and content.

## Weave API Usage Guide

### Overview
The Weave API is used in this project to query and filter evaluation traces. The API supports MongoDB-style query syntax for complex filtering operations. This guide documents the correct usage patterns and common pitfalls.

### Key Concepts

1. **Dual Filter Parameters**
   The Weave API uses two separate parameters for filtering:
   - `filter`: Basic filters like `op_names` and `parent_ids`
   - `query`: Complex MongoDB-style queries using `$expr` for field-based filtering

2. **Required Filter Fields**
   When filtering evaluation children, you must include:
   - `op_names`: The operation name(s) to filter by
   - `parent_ids`: The parent evaluation ID(s)

3. **Boolean Value Handling**
   Boolean values in `$literal` expressions must be converted to lowercase strings:
   - `True` → `"true"`
   - `False` → `"false"`

### Correct Filter Syntax

```python
# Example: Filter for output.scores.affiliation_score.correct == False
calls = client.get_calls(
    filter={
        "op_names": ["weave:///wandb-applied-ai-team/eval-failures/op/Evaluation.predict_and_score:..."],
        "parent_ids": ["0197a72d-2704-7ced-8c07-0fa1e0ab0557"]
    },
    query={
        "$expr": {
            "$eq": [
                {"$getField": "output.scores.affiliation_score.correct"},
                {"$literal": "false"}  # Note: boolean as string
            ]
        }
    }
)
```

### Implementation in weave_query.py

The `query_evaluation_children` method in `fails/weave_query.py` implements this pattern:

```python
def query_evaluation_children(self, eval_id, filter_dict=None, ...):
    # Build filter with required fields
    api_filter = {"parent_ids": [eval_id]}
    
    # Get op_name from a sample child
    sample_children = self.query_evaluation_children_helper(eval_id, limit=1)
    if sample_children:
        api_filter["op_names"] = [sample_children[0].op_name]
    
    # Build MongoDB-style query for field filtering
    if filter_dict:
        expr_conditions = []
        for field, value in filter_dict.items():
            # Convert boolean to string for $literal
            literal_value = str(value).lower() if isinstance(value, bool) else value
            expr_conditions.append({
                "$eq": [
                    {"$getField": field},
                    {"$literal": literal_value}
                ]
            })
        
        # Use $and for multiple conditions
        if len(expr_conditions) == 1:
            query = {"$expr": expr_conditions[0]}
        else:
            query = {"$expr": {"$and": expr_conditions}}
```

### Common Pitfalls and Solutions

1. **Missing op_names Filter**
   - Problem: Filtering without `op_names` returns incorrect results
   - Solution: Always include the operation name in the filter

2. **Boolean Values Not Working**
   - Problem: Using `{"$literal": False}` doesn't match boolean fields
   - Solution: Convert to string: `{"$literal": "false"}`

3. **Incorrect Filter Structure**
   - Problem: Using old-style filters like `{"op": "EqOperation", "field": field, "value": value}`
   - Solution: Use MongoDB-style `$expr` queries as shown above

4. **Client-Side vs Server-Side Filtering**
   - Server-side filtering (using Weave API) is preferred for performance
   - Client-side filtering should only be used for complex conditions not supported by the API

### MongoDB Query Operators

The Weave API supports standard MongoDB query operators:
- `$expr`: Allows field comparisons
- `$getField`: Access nested fields using dot notation
- `$literal`: Literal values for comparison
- `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`: Comparison operators
- `$and`, `$or`: Logical operators
- `$in`, `$nin`: Array membership operators

### Testing Filter Queries

When debugging filter issues:
1. First fetch unfiltered data to understand the structure
2. Test with simple filters before complex ones
3. Verify boolean values are strings in `$literal`
4. Check that op_names are included in the filter

### Example Usage in Pipeline

In `fails/pipeline.py`, the Weave API is used to filter evaluation traces:

```python
# Configuration specifies the filter
failure_config = {
    "failure_column": "output.scores.affiliation_score.correct",
    "failure_value": False
}

# WeaveQuery handles the API call with proper filter syntax
weave_query = WeaveQuery(project_name)
failure_traces = weave_query.query_evaluation_children(
    eval_id=evaluation_id,
    filter_dict={
        failure_config["failure_column"]: failure_config["failure_value"]
    },
    limit=sample_limit
)
```

This approach ensures efficient server-side filtering while maintaining compatibility with the Weave API's requirements.

## Error Handling and Robustness in weave_query.py

### Current Error Handling

The `weave_query.py` module includes basic error handling:
- HTTP errors are caught and re-raised with status codes
- JSON decode errors are logged but don't stop processing
- Missing API keys raise ValueError
- Network timeouts are configurable

### Recommended Improvements

1. **Add Retry Logic for Transient Failures**
   ```python
   @retry_on_failure(max_retries=3, backoff_factor=1.0)
   def _make_api_request(self, ...):
       # Existing request logic
   ```
   This handles temporary network issues and server errors gracefully.

2. **Custom Exception Classes**
   - `WeaveAuthenticationError` - for 401 errors
   - `WeaveProjectNotFoundError` - for 403 errors  
   - `WeaveRateLimitError` - for 429 errors
   - `WeaveNetworkError` - for connection issues

3. **Connection Pooling**
   Use `requests.Session()` for better performance:
   ```python
   self.session = requests.Session()
   adapter = requests.adapters.HTTPAdapter(
       pool_connections=10,
       pool_maxsize=10
   )
   self.session.mount('https://', adapter)
   ```

4. **Graceful Degradation**
   If server-side filtering fails, fall back to client-side:
   ```python
   try:
       return query_with_filter()
   except WeaveQueryError:
       console.print("[yellow]Falling back to client-side filtering[/yellow]")
       return client_side_filter(query_without_filter())
   ```

5. **Rate Limit Handling**
   - Check for 429 status codes
   - Extract `Retry-After` header
   - Implement exponential backoff

6. **Health Check Method**
   ```python
   def health_check(self) -> bool:
       """Verify API connectivity and credentials."""
       try:
           self._execute_query({"project_id": "...", "limit": 1})
           return True
       except:
           return False
   ```

### Error Context Best Practices

1. **Include Context in Errors**
   - Project/entity names
   - Operation being performed
   - Relevant parameters

2. **Log Warnings for Recoverable Errors**
   - JSON parse failures on individual lines
   - Fallback to client-side filtering
   - Retry attempts

3. **Fail Fast for Configuration Errors**
   - Missing API keys
   - Invalid project/entity names
   - Malformed queries

These improvements make the Weave query module more robust and production-ready while maintaining backward compatibility.
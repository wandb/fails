# CLAUDE.md - Project Knowledge Base

## UI/UX Style Guide

### Color Scheme
The Fails project uses a consistent color scheme across all terminal interfaces for visual hierarchy and clarity:

#### Primary Colors
- **Bright Cyan** (`\033[96m` or `[bright_cyan]`): 
  - Step headers and navigation (e.g., "Step 1: Enter Weave Evaluation URL")
  - Selected items in terminal selectors
  - Important column/field values
  - Panel borders for main content

- **Bright Magenta** (`\033[95m` or `[bright_magenta]`):
  - Success indicators (always with space: `✓ ` not `✓`)
  - Completion messages
  - Positive outcomes
  - The FAILS logo border
  - Group headers in selectors (bold)

- **Yellow** (`[yellow]`):
  - Warnings and important notices
  - Configuration changes
  - Error recovery suggestions

#### Secondary Colors
- **Grey/Dim** (`[dim]` or `\033[2m`):
  - Meta information (timestamps, IDs)
  - Keyboard shortcuts and instructions
  - "Using saved preferences" type messages
  - Less important status updates
  - File paths in informational messages

- **Green** (not bright) (`[green]`):
  - File save confirmations (path portion)
  - Subtle positive states
  - Secondary confirmations

- **White** (default):
  - Main content text
  - User data
  - Primary information

### Visual Hierarchy
```
1. bright_magenta - Success/completion (demands attention)
2. bright_cyan - Current step/actions (guides flow)  
3. yellow - Warnings/changes (needs awareness)
4. white - Main content (default reading)
5. green - Confirmations/saves (subtle positive)
6. dim/grey - Supporting info (optional reading)
```

### UI Patterns

#### Success Messages
Always format with space after checkmark and bright_magenta:
```python
# Good
console.print("[bright_magenta]✓ Task completed successfully![/bright_magenta]")

# Bad
console.print("[green]✓Task completed[/green]")
```

#### File Operations
Split checkmark and path with different colors:
```python
# Good
console.print(f"[bright_magenta]✓[/bright_magenta] [green]Saved to {filepath}[/green]")

# Bad  
console.print(f"[bright_magenta]✓ Saved to {filepath}[/bright_magenta]")
```

#### Step Headers
Use Panels with cyan borders for pipeline steps:
```python
console.print(
    Panel(
        "Starting task description...",
        title="Step 1: Task Name",
        border_style="cyan",
        padding=(0, 1),
    )
)
```

#### Interactive Selection Headers
Bold cyan for interactive step headers:
```python
console.print("\n[bold cyan]Step 1: Enter Weave Evaluation URL[/bold cyan]")
```

#### Status Messages
Use dim for less important status updates:
```python
# Good
console.print("[dim]Fetching evaluation trace...[/dim]")
console.print("[dim]Using discovered columns for selection...[/dim]")

# Bad
console.print("Fetching evaluation trace...")
```

### Terminal UI Consistency

#### Step Numbering
Main flow uses consistent numbering:
- Step 1: Enter Weave Evaluation URL
- Step 2: Provide Context (AI system and evaluation descriptions)
- Step 3: Select Failure Filter (with sub-steps 3a, 3b, 3c, 3d)
  - 3a: Select the column
  - 3b: Choose the operator
  - 3c: Enter the value
  - 3d: Confirm the filter (ensures it identifies INCORRECT samples)
- Step 4: Select Context Columns

#### Box Styles
- Use rounded boxes (`╭─╮`) for raw terminal UIs
- Use Rich Panels for reports and summaries
- Consistent padding and alignment

#### Text Alignment
- Success indicators: Left-aligned with 2-space indent
- Headers: Centered in boxes where appropriate
- Instructions: Grey/dim text at bottom of boxes

## Terminal UI Development Guide

This project has three main TUI implementations:
1. **Raw terminal mode selectors** (`column_context_selector.py`, `failure_selector.py`) - Arrow-based selection
2. **prompt_toolkit implementation** (`evaluation_selector.py`) - Modern bordered input with paste protection

## Terminal UI Arrow Selector Implementation (Raw Mode)

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

## prompt_toolkit Implementation (Modern Approach)

### Why prompt_toolkit?

During development of the evaluation_selector.py, we encountered severe issues with raw terminal mode when handling large paste operations:

1. **Terminal Freezing**: Large pastes (>500 chars) would freeze the terminal and crash the IDE
2. **Escape Sequence Leakage**: Bracketed paste end sequences (`[201~`) would leak into user input
3. **Repeated Error Messages**: Large pastes were processed in chunks, each triggering separate errors
4. **Poor User Experience**: No visual feedback during paste rejection

### The prompt_toolkit Solution

prompt_toolkit provides a robust framework that handles all these edge cases automatically:

```python
from prompt_toolkit import Application
from prompt_toolkit.widgets import Frame, TextArea
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.formatted_text import FormattedText
```

### Key Features Implemented

1. **Automatic Paste Handling**
   - prompt_toolkit handles bracketed paste mode automatically
   - No manual escape sequence consumption needed
   - Clean rejection of oversized pastes without terminal artifacts

2. **Bordered Input Box**
   ```python
   input_frame = Frame(
       text_area,
       title="Enter ID/URL",
       width=75,  # Fixed width
       height=5,  # Fixed height
   )
   ```

3. **Multi-line Text Area with Wrapping**
   ```python
   text_area = TextArea(
       multiline=True,   # Allow multiline display
       wrap_lines=True,  # Wrap long lines
       height=3,         # Display up to 3 lines
   )
   ```

4. **Real-time Input Validation**
   ```python
   def on_text_changed(_):
       if len(text_area.text) > MAX_INPUT_SIZE:
           text_area.text = ""  # Clear immediately
           self.error_message = "❌ Input too long..."
   
   text_area.buffer.on_text_changed += on_text_changed
   ```

5. **Styled Text with FormattedText**
   ```python
   header_text = FormattedText([
       ('', 'Normal text\n'),
       ('class:instructions', 'Grey instruction text\n'),
   ])
   
   style = Style.from_dict({
       'instructions': 'fg:#888888',  # Grey color
       'error': 'fg:red bold',
   })
   ```

6. **Left-aligned Layout with VSplit**
   ```python
   # Left-align a fixed-width frame
   left_aligned_input = VSplit([
       input_frame,      # Fixed width component
       Window(),         # Expands to fill remaining space
   ])
   ```

### Layout Best Practices

1. **Use HSplit for Vertical Stacking**
   ```python
   root_container = HSplit([
       Window(header_control, height=14),  # Fixed height header
       Window(height=1),                    # Breathing room
       left_aligned_input,                  # Input area
       error_window,                        # Conditional error display
   ])
   ```

2. **Conditional Containers**
   ```python
   ConditionalContainer(
       Window(error_control, height=3),
       filter=Condition(lambda: bool(self.error_message))
   )
   ```
   Only shows the error window when there's an error message.

3. **FormattedTextControl vs Label**
   - Use `FormattedTextControl` for dynamic text that changes
   - Wrap it in a `Window` for proper display
   - Avoid using `Label` widget with lambda functions (causes 'reset' errors)

### Key Differences from Raw Terminal Mode

| Aspect | Raw Terminal Mode | prompt_toolkit |
|--------|------------------|----------------|
| **Paste Handling** | Manual escape sequence parsing | Automatic |
| **Input Validation** | Complex state management | Event-driven callbacks |
| **Layout** | Manual positioning with ANSI codes | Declarative container system |
| **Styling** | ANSI escape sequences | CSS-like style dictionaries |
| **Text Wrapping** | Manual calculation | Built-in support |
| **Scrolling** | Manual implementation | Automatic with scrollbar option |
| **Mouse Support** | Manual escape sequence handling | Built-in with `mouse_support=True` |

### When to Use Each Approach

**Use Raw Terminal Mode when:**
- Building simple selection interfaces
- Need minimal dependencies
- Want full control over every pixel
- Performance is critical
- Examples: column_context_selector.py, failure_selector.py

**Use prompt_toolkit when:**
- Handling text input from users
- Need robust paste protection
- Want professional-looking borders and styling
- Building complex forms or dialogs
- Need mouse support
- Example: evaluation_selector.py

### Common Pitfalls and Solutions

1. **Problem: AttributeError: 'Label' object has no attribute 'reset'**
   - Cause: Using Label widget with dynamic text function
   - Solution: Use FormattedTextControl instead:
   ```python
   # Wrong
   Label(lambda: self.error_message)
   
   # Right
   FormattedTextControl(text=lambda: self.error_message)
   ```

2. **Problem: Frame border extends full width while content is narrower**
   - Cause: Frame naturally expands to fill available space
   - Solution: Wrap in VSplit with empty Window:
   ```python
   VSplit([
       Frame(..., width=75),  # Fixed width
       Window(),              # Fills remaining space
   ])
   ```

3. **Problem: No breathing room between UI elements**
   - Solution: Add empty Windows with fixed height:
   ```python
   HSplit([
       header_window,
       Window(height=1),  # Breathing room
       input_frame,
   ])
   ```

### Testing Paste Protection

To test paste protection robustly:

1. Generate large test data:
   ```bash
   python -c "print('x' * 1000)" | pbcopy  # macOS
   python -c "print('x' * 1000)" | xclip   # Linux
   ```

2. Test scenarios:
   - Paste data larger than MAX_INPUT_SIZE
   - Paste multi-line content
   - Paste binary or special characters
   - Rapid repeated pastes

3. Expected behavior:
   - Input clears immediately
   - Single error message appears
   - No terminal artifacts or escape sequences
   - Terminal remains responsive

### Key Takeaways from TUI Development

1. **Always Test Edge Cases**
   - Large paste operations can break raw terminal input
   - Escape sequences need complete consumption
   - Different terminals behave differently

2. **Choose the Right Tool**
   - Raw terminal mode: Great for simple navigation
   - prompt_toolkit: Essential for text input and complex UIs
   - Rich: Excellent for styled output but limited for input

3. **User Experience Matters**
   - Clear visual feedback for all actions
   - Proper error messages without terminal artifacts
   - Consistent keyboard shortcuts across all selectors
   - Professional appearance with borders and proper alignment

4. **Robustness Requirements**
   - Handle paste operations gracefully
   - Prevent terminal freezing at all costs
   - Clean up terminal state in finally blocks
   - Test with various terminal emulators

5. **Visual Design Consistency**
   - Use the same symbols across all selectors (▶ for cursor, ✓ for selected)
   - Consistent color scheme (cyan for highlights, grey for instructions)
   - Group related items with headers
   - Provide breathing room between sections

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

## Enhanced Filtering Capabilities in weave_query.py

The `query_evaluation_children` method now supports comprehensive filtering options beyond simple boolean values. This enables powerful server-side filtering using MongoDB-style query syntax.

### Supported Filter Types

1. **Basic Value Filtering**
   - **Booleans**: `{"field": False}` - Converted to lowercase strings ("true"/"false")
   - **Floats**: `{"field": 3.14}` - Uses `$convert` to "double" for comparison
   - **Integers**: `{"field": 42}` - Uses `$convert` to "int" for comparison
   - **Strings**: `{"field": "value"}` - Direct comparison
   - **Null values**: `{"field": None}` - Direct comparison
   - **Empty strings**: `{"field": ""}` - Direct comparison

2. **String Operations**
   - **Contains**: `{"field": {"$contains": "substring"}}`
   - **Not Contains**: `{"field": {"$not_contains": "substring"}}`

3. **Comparison Operators**
   - **Greater Than**: `{"field": {"$gt": value}}`
   - **Greater or Equal**: `{"field": {"$gte": value}}`
   - **Less Than**: `{"field": {"$lt": value}}`
   - **Less or Equal**: `{"field": {"$lte": value}}`
   - **Not Equal**: `{"field": {"$ne": value}}`

4. **Array Operations**
   - **In Array**: `{"field": {"$in": [value1, value2]}}`
   - **Not In Array**: `{"field": {"$nin": [value1, value2]}}`

5. **Existence Checks**
   - **Field Exists**: `{"field": {"$exists": True}}`
   - **Field Not Exists**: `{"field": {"$exists": False}}`

6. **Date/Timestamp Filtering**
   - Timestamps (started_at, ended_at) work with comparison operators without conversion
   - Example: `{"started_at": {"$gte": 1752500000.0}}`

### Usage Examples

```python
from fails.weave_query import WeaveQueryConfig, WeaveQueryClient

# Create client
config = WeaveQueryConfig(
    wandb_entity="wandb-applied-ai-team",
    wandb_project="eval-failures"
)
client = WeaveQueryClient(config)

# Example 1: Filter by boolean value
traces = client.query_evaluation_children(
    evaluation_call_id="0197a72d-2704-7ced-8c07-0fa1e0ab0557",
    filter_dict={"output.scores.affiliation_score.correct": False}
)

# Example 2: Filter by float with greater than
traces = client.query_evaluation_children(
    evaluation_call_id="0197a72d-2704-7ced-8c07-0fa1e0ab0557",
    filter_dict={"output.model_latency": {"$gt": 2.0}}
)

# Example 3: Filter by string containing substring
traces = client.query_evaluation_children(
    evaluation_call_id="0197a72d-2704-7ced-8c07-0fa1e0ab0557",
    filter_dict={"output.output.affiliation": {"$contains": "ext"}}
)

# Example 4: Complex filter with multiple conditions
traces = client.query_evaluation_children(
    evaluation_call_id="0197a72d-2704-7ced-8c07-0fa1e0ab0557",
    filter_dict={
        "output.scores.correct": False,
        "output.model_latency": {"$gt": 2.0},
        "output.status": {"$in": ["completed", "success"]},
        "started_at": {"$gte": 1752500000.0}
    }
)

# Example 5: Filter for null or empty values
traces = client.query_evaluation_children(
    evaluation_call_id="0197a72d-2704-7ced-8c07-0fa1e0ab0557",
    filter_dict={
        "output.error": None,  # Find traces where error is null
        "inputs.inputs.row_output": ""  # Find traces where row_output is empty
    }
)
```

### Implementation Details

The filtering logic automatically detects the type of each filter value and applies the appropriate MongoDB query syntax:

1. **Type Detection**: The implementation checks if a value is a dict (for operators), bool, float, int, None, or string
2. **Operator Handling**: When a dict is provided as the value, it extracts the operator and operand
3. **Numeric Conversions**: Float and integer comparisons use `$convert` for proper type handling, except for timestamp fields
4. **Boolean Conversion**: Boolean values are converted to lowercase strings as required by the Weave API
5. **Multiple Conditions**: When multiple filter conditions are provided, they're combined with `$and`

This enhanced filtering capability makes it much easier to query specific subsets of evaluation traces based on any combination of conditions.


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
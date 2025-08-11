#!/usr/bin/env python3
"""
Interactive selector for choosing a failure column to filter by.
"""

import sys
import termios
import tty
from typing import List, Tuple, Optional, Dict, Any
from rich.console import Console
from rich.panel import Panel


class UserCancelledException(Exception):
    """Exception raised when user cancels the selection."""


class FailureColumnSelector:
    """Interactive selector for choosing a failure column and filter value."""
    
    # Define all available operators
    OPERATORS = {
        "equals": {"symbol": "==", "key": "$eq", "needs_value": True},
        "not_equals": {"symbol": "!=", "key": "$ne", "needs_value": True},
        "greater_than": {"symbol": ">", "key": "$gt", "needs_value": True},
        "greater_or_equal": {"symbol": ">=", "key": "$gte", "needs_value": True},
        "less_than": {"symbol": "<", "key": "$lt", "needs_value": True},
        "less_or_equal": {"symbol": "<=", "key": "$lte", "needs_value": True},
        "contains": {"symbol": "contains", "key": "$contains", "needs_value": True},
        "not_contains": {"symbol": "does not contain", "key": "$not_contains", "needs_value": True},
        "in_list": {"symbol": "in", "key": "$in", "needs_value": True},
        "not_in_list": {"symbol": "not in", "key": "$nin", "needs_value": True},
        "exists": {"symbol": "exists", "key": "$exists", "needs_value": False, "value": True},
        "not_exists": {"symbol": "does not exist", "key": "$exists", "needs_value": False, "value": False},
    }
    
    def __init__(self, columns: List[str], sample_values: Optional[Dict[str, List[Any]]] = None):
        self.columns = sorted(columns)
        self.current_index = 0
        self.selected_column: Optional[str] = None
        self.selected_operator: Optional[str] = None
        self.filter_value: Optional[Any] = None
        self.stage = "column"  # "column", "operator", "value", or "confirm"
        self.sample_values = sample_values or {}
        self.detected_type: Optional[str] = None
        self.operator_index = 0
        self.value_input = ""
        self.value_selection_index = 0  # For arrow key navigation in value selection
        
        # Group columns for better display
        self.items = []
        grouped = {}
        other = []
        
        for col in self.columns:
            if '.' in col:
                prefix = col.split('.')[0]
                if prefix not in grouped:
                    grouped[prefix] = []
                grouped[prefix].append(col)
            else:
                other.append(col)
        
        # Separate output.scores from other output columns
        output_scores = []
        output_other = []
        if 'output' in grouped:
            for col in grouped['output']:
                if col.startswith('output.scores.'):
                    output_scores.append(col)
                else:
                    output_other.append(col)
        
        # Build flat list with group headers in specific order
        # 1. Output.scores group
        if output_scores:
            self.items.append(('group', 'Output.scores'))
            for col in sorted(output_scores):
                self.items.append(('column', col))
        
        # 2. Output group (remaining columns)
        if output_other:
            self.items.append(('group', 'Output'))
            for col in sorted(output_other):
                self.items.append(('column', col))
        
        # 3. Summary group
        if 'summary' in grouped:
            self.items.append(('group', 'Summary'))
            for col in sorted(grouped['summary']):
                self.items.append(('column', col))
        
        # 4. Input group (using 'inputs' key)
        if 'inputs' in grouped:
            self.items.append(('group', 'Inputs'))
            for col in sorted(grouped['inputs']):
                self.items.append(('column', col))
        
        # 5. Attributes group
        if 'attributes' in grouped:
            self.items.append(('group', 'Attributes'))
            for col in sorted(grouped['attributes']):
                self.items.append(('column', col))
        
        # 6. All other groups alphabetically (excluding the ones we've already handled)
        handled_groups = {'output', 'summary', 'inputs', 'attributes'}
        for group in sorted(grouped.keys()):
            if group not in handled_groups:
                display_name = group.capitalize()
                self.items.append(('group', display_name))
                for col in sorted(grouped[group]):
                    self.items.append(('column', col))
        
        # 7. Other ungrouped columns
        if other:
            self.items.append(('group', "Other"))
            for col in sorted(other):
                self.items.append(('column', col))
        
        # Ensure we start on a column, not a group header
        while self.current_index < len(self.items) and self.items[self.current_index][0] == 'group':
            self.current_index += 1
    
    def format_sample_values(self, samples: List[Any], max_samples: int = 5, max_total_length: int = 200) -> str:
        """Format sample values for display with proper truncation."""
        formatted_samples = []
        max_sample_length = 50  # Max chars per individual sample
        
        for s in samples[:max_samples]:
            if s is None:
                continue
                
            if isinstance(s, dict):
                # For dicts, show a preview of keys and truncated values
                dict_parts = []
                for k, v in list(s.items())[:3]:  # Show first 3 key-value pairs
                    v_str = str(v)
                    if len(v_str) > 20:
                        v_str = v_str[:17] + "..."
                    dict_parts.append(f"'{k}': {repr(v_str)}")
                
                dict_preview = "{" + ", ".join(dict_parts)
                if len(s) > 3:
                    dict_preview += ", ..."
                dict_preview += "}"
                formatted_samples.append(dict_preview)
            else:
                # For other types, truncate if too long
                s_str = str(s)
                if len(s_str) > max_sample_length:
                    s_str = s_str[:max_sample_length-3] + "..."
                formatted_samples.append(s_str)
        
        sample_str = ", ".join(formatted_samples)
        
        # Limit total length
        if len(sample_str) > max_total_length:
            sample_str = sample_str[:max_total_length-3] + "..."
            
        if len(samples) > max_samples:
            sample_str += f" (+{len(samples) - max_samples} more)"
            
        return sample_str
    
    def detect_type(self, column: str) -> str:
        """Detect the data type from sample values."""
        if column not in self.sample_values or not self.sample_values[column]:
            return "unknown"
        
        samples = self.sample_values[column]
        
        # Check for booleans
        if all(isinstance(v, bool) or v in ["true", "false", "True", "False"] for v in samples if v is not None):
            return "boolean"
        
        # Check for numbers
        if all(isinstance(v, (int, float)) or (isinstance(v, str) and v.replace('.', '').replace('-', '').isdigit()) for v in samples if v is not None):
            return "numeric"
        
        # Check for dates/timestamps
        if all(isinstance(v, (int, float)) and v > 1000000000 and v < 2000000000 for v in samples if v is not None):
            return "timestamp"
        
        # Default to string
        return "string"
    
    def get_recommended_operators(self, data_type: str) -> List[str]:
        """Get recommended operators based on data type."""
        if data_type == "boolean":
            return ["equals", "not_equals", "exists", "not_exists"]
        elif data_type == "numeric":
            return ["equals", "not_equals", "greater_than", "greater_or_equal", "less_than", "less_or_equal"]
        elif data_type == "timestamp":
            return ["greater_than", "greater_or_equal", "less_than", "less_or_equal", "equals"]
        elif data_type == "string":
            return ["equals", "not_equals", "contains", "not_contains", "in_list", "not_in_list"]
        else:
            return ["equals", "not_equals", "exists", "not_exists"]
    
    def run(self) -> Tuple[Optional[str], Optional[Any]]:
        """Run the selector and return (column, filter_value)."""
        # Check if we're in an interactive terminal
        if not sys.stdin.isatty():
            raise RuntimeError("Interactive selection requires a TTY. Use --no-interactive flag or provide filter via command line.")
        
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        
        try:
            # Enter alternate screen buffer and hide cursor
            sys.stdout.write("\033[?1049h\033[?25l")
            sys.stdout.flush()
            
            # Set terminal to raw mode
            tty.setraw(sys.stdin.fileno())
            
            # Initial clear
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()
            
            while True:
                if self.stage == "column":
                    if not self._display_column_selection():
                        break
                elif self.stage == "operator":
                    if not self._display_operator_selection():
                        break
                elif self.stage == "value":
                    if not self._display_value_selection():
                        break
                elif self.stage == "confirm":
                    if not self._display_confirmation():
                        break
                else:
                    break
        except UserCancelledException:
            # Clean up terminal before re-raising
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            sys.stdout.write("\033[?25h\033[?1049l")
            sys.stdout.flush()
            raise
        finally:
            # Restore terminal
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            sys.stdout.write("\033[?25h\033[?1049l")
            sys.stdout.flush()
        
        # Build return value based on selected operator
        if self.selected_column and self.selected_operator:
            op_info = self.OPERATORS[self.selected_operator]
            if not op_info["needs_value"]:
                # For exists/not_exists operators
                return self.selected_column, {op_info["key"]: op_info["value"]}
            elif self.filter_value is not None:
                # For operators that need values
                if self.selected_operator == "equals":
                    # Direct value for equality
                    return self.selected_column, self.filter_value
                else:
                    # Operator format for other operators
                    return self.selected_column, {op_info["key"]: self.filter_value}
        
        return None, None
    
    def _display_column_selection(self) -> bool:
        """Display column selection interface. Returns False to exit."""
        # Move to home
        sys.stdout.write("\033[H")
        
        def write_line(content="", clear=True):
            """Helper function to write a line with optional clearing"""
            prefix = "\033[K" if clear else ""
            sys.stdout.write(f"{prefix}{content}\r\n")

        write_line()  # Empty line
        write_line()  # Empty line

        # Header with rounded box
        write_line()
        write_line("  ╭" + "─" * 92 + "╮")
        write_line("  │  \033[1;96mStep 3: Create Failure Filter\033[0m" + " " * 61 + "│")
        write_line("  │" + " " * 92 + "│")
        write_line("  │  \033[1m3a. Column Selection\033[0m" + " " * 70 + "│")
        write_line("  │" + " " * 92 + "│")
        write_line("  │  Select the 1 column that indicates there was an evaluation failure." + " " * 23 + "│")
        write_line("  │  - This should correspond to a column name in your Weave traces table in the app." + " " * 10 + "│")
        write_line("  │  - This could be a boolean, numeric or string column like 'is_correct' or 'score' etc." + " " * 5 + "│")
        write_line("  │  - The failure categorization will use this column to find the failures to focus on." + " " * 7 + "│")
        write_line("  │" + " " * 92 + "│")
        write_line("  │  \033[2m↑/↓: Navigate    Space: Select    q: Cancel\033[0m" + " " * 47 + "│")
        write_line("  ╰" + "─" * 92 + "╯")
        write_line()
        write_line()
        
        # Calculate window
        window_size = 20
        half_window = window_size // 2
        
        if len(self.items) <= window_size:
            start = 0
            end = len(self.items)
        elif self.current_index < half_window:
            start = 0
            end = window_size
        elif self.current_index >= len(self.items) - half_window:
            end = len(self.items)
            start = max(0, end - window_size)
        else:
            start = self.current_index - half_window
            end = self.current_index + half_window + 1
        
        # Show scroll indicator
        if start > 0:
            sys.stdout.write(f"\033[K    ↑ {start} more above ↑\r\n")
            sys.stdout.write("\033[K\r\n")
        
        # Display items
        for i in range(start, end):
            if i >= len(self.items):
                break
                
            item_type, item_data = self.items[i]
            is_current = i == self.current_index
            
            if item_type == 'group':
                if i > 0 and i > start:
                    sys.stdout.write("\033[K\r\n")
                # Group headers are never selectable, always use same prefix
                sys.stdout.write(f"\033[K   \033[1;95m{item_data}\033[0m\r\n")
            else:
                # Column
                if is_current:
                    prefix = " ▶ "
                    # Highlight current selection
                    sys.stdout.write(f"\033[K{prefix} \033[96m{item_data}\033[0m\r\n")
                else:
                    prefix = "   "
                    sys.stdout.write(f"\033[K{prefix}   {item_data}\r\n")
        
        # Show scroll indicator
        if end < len(self.items):
            sys.stdout.write("\033[K\r\n")
            sys.stdout.write(f"\033[K    ↓ {len(self.items) - end} more below ↓\r\n")
        
        sys.stdout.write("\033[K\r\n")
        sys.stdout.write("\033[K" + "─" * 90 + "\r\n")
        
        # Clear rest of screen
        sys.stdout.write("\033[J")
        sys.stdout.flush()
        
        # Read key
        key = sys.stdin.read(1)
        
        # Handle arrow keys
        if key == '\x1b':
            next_chars = sys.stdin.read(2)
            if next_chars == '[A':  # Up
                self.current_index = max(0, self.current_index - 1)
                # Skip group headers when navigating
                while (self.current_index > 0 and 
                       self.items[self.current_index][0] == 'group'):
                    self.current_index -= 1
            elif next_chars == '[B':  # Down
                self.current_index = min(len(self.items) - 1, self.current_index + 1)
                # Skip group headers when navigating
                while (self.current_index < len(self.items) - 1 and 
                       self.items[self.current_index][0] == 'group'):
                    self.current_index += 1
            return True
        
        # Handle other keys
        if key == 'q':
            # Raise exception when user cancels instead of exiting
            raise UserCancelledException("Selection cancelled by user")
        elif key == '\r' or key == '\n' or key == ' ':  # Enter or Space
            if self.items[self.current_index][0] == 'column':
                self.selected_column = self.items[self.current_index][1]
                self.detected_type = self.detect_type(self.selected_column)
                self.stage = "operator"
                self.operator_index = 0  # Reset operator selection
                return True
        elif key in ['j', 'k']:  # Vim keys
            if key == 'j':
                self.current_index = min(len(self.items) - 1, self.current_index + 1)
                while (self.current_index < len(self.items) - 1 and 
                       self.items[self.current_index][0] == 'group'):
                    self.current_index += 1
            else:
                self.current_index = max(0, self.current_index - 1)
                while (self.current_index > 0 and 
                       self.items[self.current_index][0] == 'group'):
                    self.current_index -= 1
        
        return True
    
    def _display_operator_selection(self) -> bool:
        """Display operator selection interface. Returns False to exit."""
        # Move to home
        sys.stdout.write("\033[H")
        
        def write_line(content="", clear=True):
            """Helper function to write a line with optional clearing"""
            prefix = "\033[K" if clear else ""
            sys.stdout.write(f"{prefix}{content}\r\n")
        
        # Header
        write_line()
        write_line()
        write_line("  ╭" + "─" * 92 + "╮")
        write_line("  │  \033[1;96mStep 3: Create Failure Filter\033[0m" + " " * 61 + "│")
        write_line("  │" + " " * 92 + "│")
        write_line("  │  \033[1m3b. Operator Selection\033[0m" + " " * 67 + "│")
        write_line("  │" + " " * 92 + "│")
        write_line("  │  Choose how to compare values for your selected column" + " " * 37 + "│")
        write_line("  │  \033[2m- You will select the value to compare against next\033[0m" + " " * 39 + "│")
        write_line("  │" + " " * 92 + "│")
        write_line("  │  \033[2m↑/↓: Navigate    Space: Select    b: Back\033[0m" + " " * 49 + "│")
        write_line("  ╰" + "─" * 92 + "╯")
        write_line()
        
        # Show column and sample values
        write_line(f"Selected column: \033[96m{self.selected_column}\033[0m")
        
        if self.selected_column in self.sample_values and self.sample_values[self.selected_column]:
            samples = self.sample_values[self.selected_column][:5]
            
            # Format samples with length limit
            formatted_samples = []
            max_sample_length = 50  # Max chars per individual sample
            
            for s in samples:
                if s is None:
                    continue
                    
                if isinstance(s, dict):
                    # For dicts, show a preview of keys and truncated values
                    dict_parts = []
                    for k, v in list(s.items())[:3]:  # Show first 3 key-value pairs
                        v_str = str(v)
                        if len(v_str) > 20:
                            v_str = v_str[:17] + "..."
                        dict_parts.append(f"'{k}': {repr(v_str)}")
                    
                    dict_preview = "{" + ", ".join(dict_parts)
                    if len(s) > 3:
                        dict_preview += ", ..."
                    dict_preview += "}"
                    formatted_samples.append(dict_preview)
                else:
                    # For other types, truncate if too long
                    s_str = str(s)
                    if len(s_str) > max_sample_length:
                        s_str = s_str[:max_sample_length-3] + "..."
                    formatted_samples.append(s_str)
            
            sample_str = ", ".join(formatted_samples)
            
            # Limit total length
            max_total_length = 200
            if len(sample_str) > max_total_length:
                sample_str = sample_str[:max_total_length-3] + "..."
                
            if len(self.sample_values[self.selected_column]) > 5:
                sample_str += " (+" + str(len(self.sample_values[self.selected_column]) - 5) + " more)"
                
            write_line(f"Sample value(s): \033[96m{sample_str}\033[0m")
            write_line(f"Detected type: \033[96m{self.detected_type}\033[0m")
        
        write_line()
        write_line("─" * 90)
        write_line()
        
        # Get recommended operators
        recommended = self.get_recommended_operators(self.detected_type)
        all_operators = list(self.OPERATORS.keys())
        
        # Show recommended operators first
        if recommended:
            write_line("\033[1;95mRecommended operators\033[0m")
            write_line()
        
        operator_list = []
        for i, (op_key, op_info) in enumerate(self.OPERATORS.items()):
            if op_key in recommended:
                operator_list.append((op_key, op_info, True))  # True = recommended
        
        # Then show other operators
        other_operators = [op for op in all_operators if op not in recommended]
        if other_operators:
            has_other_header = False
            for op_key in other_operators:
                if not has_other_header:
                    operator_list.append(("HEADER", None, False))
                    has_other_header = True
                operator_list.append((op_key, self.OPERATORS[op_key], False))
        
        # Display operators with scrolling
        window_size = 15
        half_window = window_size // 2
        
        if len(operator_list) <= window_size:
            start = 0
            end = len(operator_list)
        elif self.operator_index < half_window:
            start = 0
            end = window_size
        elif self.operator_index >= len(operator_list) - half_window:
            end = len(operator_list)
            start = max(0, end - window_size)
        else:
            start = self.operator_index - half_window
            end = self.operator_index + half_window + 1
        
        # Show operators
        actual_index = 0
        for i in range(start, end):
            if i >= len(operator_list):
                break
            
            item = operator_list[i]
            if item[0] == "HEADER":
                write_line()
                write_line("\033[1;95mOther operators\033[0m")
                write_line()
            else:
                op_key, op_info, is_recommended = item
                is_current = i == self.operator_index
                
                # Format the operator display
                symbol = op_info["symbol"]
                if is_current:
                    prefix = " ▶ "
                    if is_recommended:
                        line = f"{prefix} \033[96m{op_key.replace('_', ' ').title()}\033[0m ({symbol})"
                    else:
                        line = f"{prefix} \033[96m{op_key.replace('_', ' ').title()}\033[0m ({symbol})"
                else:
                    prefix = "   "
                    line = f"{prefix}   {op_key.replace('_', ' ').title()} ({symbol})"
                
                write_line(line)
                actual_index += 1
        
        # Clear rest of screen
        sys.stdout.write("\033[J")
        sys.stdout.flush()
        
        # Read key
        key = sys.stdin.read(1)
        
        # Handle arrow keys
        if key == '\x1b':
            next_chars = sys.stdin.read(2)
            if next_chars == '[A':  # Up
                self.operator_index = max(0, self.operator_index - 1)
                # Skip header when going up
                while self.operator_index > 0 and operator_list[self.operator_index][0] == "HEADER":
                    self.operator_index -= 1
            elif next_chars == '[B':  # Down
                self.operator_index = min(len(operator_list) - 1, self.operator_index + 1)
                # Skip header when going down
                while self.operator_index < len(operator_list) - 1 and operator_list[self.operator_index][0] == "HEADER":
                    self.operator_index += 1
            return True
        
        # Handle other keys
        if key == 'b':
            self.stage = "column"
            self.selected_column = None
            return True
        elif key == '\r' or key == '\n' or key == ' ':  # Enter or Space
            # Get the current item from operator_list
            if self.operator_index < len(operator_list):
                item = operator_list[self.operator_index]
                if item[0] != "HEADER":
                    self.selected_operator = item[0]
                    op_info = self.OPERATORS[self.selected_operator]
                    if not op_info["needs_value"]:
                        # For exists/not_exists, go straight to confirmation
                        self.filter_value = None  # No value needed
                        self.stage = "confirm"
                        return True
                    else:
                        # Move to value input stage
                        self.stage = "value"
                        self.value_input = ""
                        self.value_selection_index = 0  # Reset for arrow navigation
                    return True
        elif key in ['j', 'k']:  # Vim keys
            if key == 'j':
                self.operator_index = min(len(operator_list) - 1, self.operator_index + 1)
                # Skip header when going down
                while self.operator_index < len(operator_list) - 1 and operator_list[self.operator_index][0] == "HEADER":
                    self.operator_index += 1
            else:
                self.operator_index = max(0, self.operator_index - 1)
                # Skip header when going up
                while self.operator_index > 0 and operator_list[self.operator_index][0] == "HEADER":
                    self.operator_index -= 1
        
        return True
    
    def _display_value_selection(self) -> bool:
        """Display value selection interface. Returns False to exit."""
        # Move to home
        sys.stdout.write("\033[H")
        
        def write_line(content="", clear=True):
            """Helper function to write a line with optional clearing"""
            prefix = "\033[K" if clear else ""
            sys.stdout.write(f"{prefix}{content}\r\n")
        
        # Header
        write_line()
        write_line()
        write_line("  ╭" + "─" * 92 + "╮")
        write_line("  │  \033[1;96mStep 3: Create Failure Filter\033[0m" + " " * 61 + "│")
        write_line("  │" + " " * 92 + "│")
        write_line("  │  \033[1m3c. Value Input\033[0m" + " " * 75 + "│")
        write_line("  │" + " " * 92 + "│")
        write_line("  │  Configure the filter value for your selected column and operator." + " " * 25 + "│")
        write_line("  │" + " " * 92 + "│")
        write_line("  │  \033[2m↑/↓: Navigate    Space: Select    b: Back\033[0m" + " " * 49 + "│")
        write_line("  ╰" + "─" * 92 + "╯")
        write_line()
        
        # Show column and operator
        op_info = self.OPERATORS[self.selected_operator]
        write_line(f"Column: \033[96m{self.selected_column}\033[0m")
        write_line(f"Operator: \033[96m{self.selected_operator.replace('_', ' ').title()}\033[0m ({op_info['symbol']})")
        write_line()
        
        # Different input methods based on operator
        if self.selected_operator in ["in_list", "not_in_list"]:
            # List input
            write_line("Enter comma-separated values:")
            write_line("Example: value1, value2, value3")
            write_line()
            write_line("─" * 90)
            write_line()
            write_line(f"Values: {self.value_input}█")
            
        elif self.detected_type == "boolean" and self.selected_operator in ["equals", "not_equals"]:
            # Boolean selection with arrow navigation
            write_line("Select boolean value:")
            write_line()
            write_line("─" * 90)
            write_line()
            
            # Display boolean options with selection indicator
            options = [(False, "False"), (True, "True")]
            for i, (value, display) in enumerate(options):
                if i == self.value_selection_index:
                    write_line(f" ▶  \033[96m{display}\033[0m")
                else:
                    write_line(f"    {display}")
            
        else:
            # Text/numeric input
            input_type = "numeric value" if self.detected_type in ["numeric", "timestamp"] else "text"
            if self.selected_column in self.sample_values and self.sample_values[self.selected_column]:
                samples = self.sample_values[self.selected_column]
                sample_str = self.format_sample_values(samples, max_samples=3)
                write_line(f"Sample values: \033[96m{sample_str}\033[0m")
            write_line(f"Enter {input_type}:")
            
            write_line()
            write_line("─" * 90)
            write_line()
            write_line(f"Value: {self.value_input}█")
        
        write_line()
        write_line("─" * 90)
        
        # Clear rest of screen
        sys.stdout.write("\033[J")
        sys.stdout.flush()
        
        # Read key
        key = sys.stdin.read(1)
        
        # Handle special keys
        if key == '\x1b':  # Escape sequences
            next_chars = sys.stdin.read(2)
            # Handle arrow keys for boolean selection
            if self.detected_type == "boolean" and self.selected_operator in ["equals", "not_equals"]:
                if next_chars == '[A':  # Up
                    self.value_selection_index = max(0, self.value_selection_index - 1)
                elif next_chars == '[B':  # Down
                    self.value_selection_index = min(1, self.value_selection_index + 1)
            return True
        
        if key == 'b':
            self.stage = "operator"
            self.value_input = ""
            self.value_selection_index = 0  # Reset selection
            return True
        
        # Handle boolean selection with Space
        if self.detected_type == "boolean" and self.selected_operator in ["equals", "not_equals"]:
            if key == ' ':  # Space to submit
                self.filter_value = self.value_selection_index == 1  # 0=False, 1=True
                self.stage = "confirm"
                return True
            elif key in ['j', 'k']:  # Vim keys
                if key == 'j':
                    self.value_selection_index = min(1, self.value_selection_index + 1)
                else:
                    self.value_selection_index = max(0, self.value_selection_index - 1)
                return True
        
        # Handle text input for other cases
        elif key == '\r' or key == '\n':  # Enter to submit (need space for typing)
            if self.value_input:
                # Parse the input based on operator
                if self.selected_operator in ["in_list", "not_in_list"]:
                    # Split by comma and strip whitespace
                    values = [v.strip() for v in self.value_input.split(',') if v.strip()]
                    self.filter_value = values
                elif self.detected_type == "numeric":
                    # Try to parse as number
                    try:
                        if '.' in self.value_input:
                            self.filter_value = float(self.value_input)
                        else:
                            self.filter_value = int(self.value_input)
                    except ValueError:
                        # Keep as string if can't parse
                        self.filter_value = self.value_input
                else:
                    # Keep as string
                    self.filter_value = self.value_input
                self.stage = "confirm"
                return True
        
        elif key == '\x7f' or key == '\b':  # Backspace
            if self.value_input:
                self.value_input = self.value_input[:-1]
        
        elif key.isprintable():
            self.value_input += key
        
        return True
    
    def _display_confirmation(self) -> bool:
        """Display filter confirmation interface. Returns False when confirmed."""
        # Move to home
        sys.stdout.write("\033[H")
        
        def write_line(content="", clear=True):
            """Helper function to write a line with optional clearing"""
            prefix = "\033[K" if clear else ""
            sys.stdout.write(f"{prefix}{content}\r\n")
        
        # Header
        write_line()
        write_line()
        write_line("  ╭" + "─" * 92 + "╮")
        write_line("  │  \033[1;96mStep 3: Create Failure Filter\033[0m" + " " * 61 + "│")
        write_line("  │" + " " * 92 + "│")
        write_line("  │  \033[1m3d. Confirm Filter\033[0m" + " " * 72 + "│")
        write_line("  │" + " " * 92 + "│")
        write_line("  │  Please confirm this filter will find the INCORRECT samples from your evaluation:" + " " * 10 + "│")
        write_line("  │" + " " * 92 + "│")
        write_line("  │  \033[2mEnter: Confirm    b: Back\033[0m" + " " * 65 + "│")
        write_line("  ╰" + "─" * 92 + "╯")
        write_line()
        write_line()
        
        # Build the filter display string
        op_info = self.OPERATORS[self.selected_operator]
        
        # Format the filter expression
        if self.selected_operator == "equals":
            if isinstance(self.filter_value, bool):
                filter_str = f"{self.selected_column} == {self.filter_value}"
            else:
                filter_str = f"{self.selected_column} == {repr(self.filter_value)}"
        elif self.selected_operator == "not_equals":
            if isinstance(self.filter_value, bool):
                filter_str = f"{self.selected_column} != {self.filter_value}"
            else:
                filter_str = f"{self.selected_column} != {repr(self.filter_value)}"
        elif self.selected_operator in ["exists", "not_exists"]:
            filter_str = f"{self.selected_column} {op_info['symbol']}"
        elif self.selected_operator in ["in_list", "not_in_list"]:
            filter_str = f"{self.selected_column} {op_info['symbol']} {self.filter_value}"
        elif self.selected_operator in ["contains", "not_contains"]:
            filter_str = f"{self.selected_column} {op_info['symbol']} {repr(self.filter_value)}"
        else:
            # Comparison operators
            filter_str = f"{self.selected_column} {op_info['symbol']} {self.filter_value}"
        
        # Display the filter in a box
        write_line("  ╭" + "─" * 92 + "╮")
        write_line("  │" + " " * 92 + "│")
        
        # Center the filter string (max 90 chars with padding)
        filter_display = f"  \033[1;96m{filter_str}\033[0m"
        # Calculate padding needed
        # Remove ANSI codes for length calculation
        clean_filter = filter_str
        padding_needed = 92 - len(clean_filter) - 4  # 4 for the "  " prefix
        left_padding = padding_needed // 2
        right_padding = padding_needed - left_padding
        
        write_line(f"  │  {' ' * left_padding}\033[1;96m{filter_str}\033[0m{' ' * right_padding}  │")
        write_line("  │" + " " * 92 + "│")
        write_line("  ╰" + "─" * 92 + "╯")
        write_line()
        write_line()
        
        # Additional context
        write_line("  \033[2mThis filter will be used to identify evaluation failures that need categorization.\033[0m")
        write_line("  \033[2mMake sure this correctly identifies the INCORRECT/FAILED samples in your evaluation.\033[0m")
        
        # Clear rest of screen
        sys.stdout.write("\033[J")
        sys.stdout.flush()
        
        # Read key
        key = sys.stdin.read(1)
        
        # Handle keys
        if key == 'b':
            # Go back to value selection (or operator if no value needed)
            if self.OPERATORS[self.selected_operator]["needs_value"]:
                self.stage = "value"
                # Reset value
                if isinstance(self.filter_value, bool):
                    self.value_selection_index = 1 if self.filter_value else 0
                elif isinstance(self.filter_value, list):
                    self.value_input = ", ".join(str(v) for v in self.filter_value)
                else:
                    self.value_input = str(self.filter_value) if self.filter_value else ""
            else:
                self.stage = "operator"
            return True
        elif key == '\r' or key == '\n':  # Enter - confirm
            return False  # Exit the loop, filter is confirmed
        
        return True


def interactive_failure_column_selection(
    console: Console,
    columns: List[str],
    sample_values: Optional[Dict[str, List[Any]]] = None
) -> Tuple[Optional[str], Optional[Any]]:
    """
    Interactive selection for failure column and filter value.
    
    Args:
        console: Rich console for display
        columns: List of available columns
        sample_values: Optional dict mapping column names to sample values
    
    Returns:
        Tuple of (column_name, filter_value) or (None, None) if cancelled
        filter_value can be:
        - A direct value (for equals operator)
        - A dict like {"$gt": 5} (for other operators)
        - None if cancelled
    """
    # Run the selector without initial message
    selector = FailureColumnSelector(columns, sample_values)
    try:
        column, value = selector.run()
    except UserCancelledException:
        console.print("[yellow]Selection cancelled by user.[/yellow]")
        return None, None
    
    # Don't show results here - they'll be shown in the Configuration Summary
    if not column or value is None:
        console.print(Panel(
            "[bold yellow]Selection Cancelled[/bold yellow]",
            border_style="yellow"
        ))
    
    return column, value 
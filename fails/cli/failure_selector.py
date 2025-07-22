#!/usr/bin/env python3
"""
Interactive selector for choosing a failure column to filter by.
"""

import sys
import termios
import tty
from typing import List, Tuple, Optional
from rich.console import Console
from rich.panel import Panel


class FailureColumnSelector:
    """Interactive selector for choosing a failure column and filter value."""
    
    def __init__(self, columns: List[str]):
        self.columns = sorted(columns)
        self.current_index = 0
        self.selected_column: Optional[str] = None
        self.filter_value: Optional[bool] = None
        self.stage = "column"  # "column" or "value"
        
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
    
    def run(self) -> Tuple[Optional[str], Optional[bool]]:
        """Run the selector and return (column, filter_value)."""
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
                elif self.stage == "value":
                    if not self._display_value_selection():
                        break
                else:
                    break
                    
        finally:
            # Restore terminal
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            sys.stdout.write("\033[?25h\033[?1049l")
            sys.stdout.flush()
        
        return self.selected_column, self.filter_value
    
    def _display_column_selection(self) -> bool:
        """Display column selection interface. Returns False to exit."""
        # Move to home
        sys.stdout.write("\033[H")
        
        # Header
        sys.stdout.write("\033[K\r\n")
        sys.stdout.write("\033[K\r\n")
        sys.stdout.write("\033[K" + "=" * 90 + "\r\n")
        sys.stdout.write("\033[K\r\n")
        sys.stdout.write("\033[K\033[1mEvaluation Failure Categorization\033[0m\r\n")
        sys.stdout.write("\033[K\r\n")
        sys.stdout.write("\033[K" + "=" * 90 + "\r\n")
        sys.stdout.write("\033[K\r\n")
        sys.stdout.write("\033[K\033[1mFailure Column Selection\033[0m\r\n")
        sys.stdout.write("\033[K" + "-" * 90 + "\r\n")
        sys.stdout.write("\033[KPlease select the column that indicates a sample failed the evaluation.\r\n")
        sys.stdout.write("\033[KThe evaluation traces will be filtered based on the value of this column.\r\n")
        sys.stdout.write("\033[K\r\n")
        sys.stdout.write("\033[K↑/↓: Navigate  |  Space: Select  |  q: Cancel\r\n")
        sys.stdout.write("\033[K" + "─" * 90 + "\r\n")
        sys.stdout.write("\033[K\r\n")
        
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
                prefix = " > " if is_current else "   "
                sys.stdout.write(f"\033[K{prefix}\033[1m{item_data}\033[0m\r\n")
            else:
                # Column
                if is_current:
                    prefix = " ▶ "
                    # Highlight current selection
                    sys.stdout.write(f"\033[K{prefix} \033[96m{item_data}\033[0m\r\n")
                else:
                    prefix = "   "
                    sys.stdout.write(f"\033[K{prefix} {item_data}\r\n")
        
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
            # Exit the entire program when user cancels
            sys.stdout.write("\033[?25h\033[?1049l")  # Show cursor and exit alternate screen
            sys.stdout.flush()
            print("Selection cancelled by user. Exiting...")
            sys.exit(0)
        elif key == '\r' or key == '\n' or key == ' ':  # Enter or Space
            if self.items[self.current_index][0] == 'column':
                self.selected_column = self.items[self.current_index][1]
                self.stage = "value"
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
    
    def _display_value_selection(self) -> bool:
        """Display value selection interface. Returns False to exit."""
        # Move to home
        sys.stdout.write("\033[H")
        
        # Smart truncation for column name in options
        display_column = self.selected_column or ""
        if len(display_column) > 30:
            # Split by dots and truncate intelligently
            parts = display_column.split('.')
            truncated = ""
            # Start from the end and work backwards
            for i in range(len(parts) - 1, -1, -1):
                if i == len(parts) - 1:
                    # Always include the last part
                    truncated = parts[i]
                else:
                    # Check if adding this part would exceed limit
                    test = parts[i] + "." + truncated
                    if len("..." + test) <= 30:
                        truncated = test
                    else:
                        break
            display_column = "..." + truncated
        
        # Header
        sys.stdout.write("\033[K" + "=" * 90 + "\r\n")
        sys.stdout.write("\033[K\033[1mValue to filter by\033[0m\r\n")
        sys.stdout.write("\033[K" + "-" * 90 + "\r\n")
        full_column = self.selected_column or ""
        sys.stdout.write(f"\033[KSelect the value for: \033[96m{full_column}\033[0m\r\n")
        sys.stdout.write("\033[Kthat indicates that the sample failed the evaluation\r\n")
        sys.stdout.write("\033[K" + "─" * 90 + "\r\n")
        sys.stdout.write("\033[K\r\n")
        sys.stdout.write("\033[K1/2: Select  |  q: Go back\r\n")
        sys.stdout.write("\033[K" + "─" * 90 + "\r\n")
        sys.stdout.write("\033[K\r\n")
        
        # Options with actual column name (using truncated version)
        # Blue for False (colorblind-friendly)
        sys.stdout.write(f"\033[K    \033[1m1)\033[0m \033[96m{display_column}\033[0m == \033[94mFalse\033[0m means the sample failed the evaluation\r\n")
        sys.stdout.write("\033[K\r\n")
        # Orange for True (colorblind-friendly) - using yellow which appears orange-ish in most terminals
        sys.stdout.write(f"\033[K    \033[1m2)\033[0m \033[96m{display_column}\033[0m == \033[93mTrue\033[0m means the sample failed the evaluation\r\n")
        sys.stdout.write("\033[K\r\n")
        sys.stdout.write("\033[K" + "─" * 90 + "\r\n")
        
        # Clear rest of screen
        sys.stdout.write("\033[J")
        sys.stdout.flush()
        
        # Read key
        key = sys.stdin.read(1)
        
        if key == 'q':
            self.stage = "column"
            return True
        elif key == '1':
            self.filter_value = False
            return False
        elif key == '2':
            self.filter_value = True
            return False
        
        return True


def interactive_failure_column_selection(
    console: Console,
    columns: List[str]
) -> Tuple[Optional[str], Optional[bool]]:
    """
    Interactive selection for failure column and filter value.
    
    Returns:
        Tuple of (column_name, filter_value) or (None, None) if cancelled
    """
    # Show initial message
    console.print(Panel(
        "[bold cyan]Failure Column Selection[/bold cyan]\n\n"
        "Select the single column that indicates a sample failed the evaluation.\n\n"
        "[dim]↑/↓: Navigate  |  Space: Select  |  q: Cancel[/dim]",
        border_style="cyan"
    ))
    
    # Run the selector
    selector = FailureColumnSelector(columns)
    column, value = selector.run()
    
    # Show results
    if column and value is not None:
        value_text = "[red]False[/red]" if value is False else "[green]True[/green]"
        console.print(Panel(
            f"[bold green]Selection Complete![/bold green]\n\n"
            f"Failure column: [cyan]{column}[/cyan]\n"
            f"Filter for: {value_text} values",
            border_style="green"
        ))
    else:
        console.print(Panel(
            "[bold yellow]Selection Cancelled[/bold yellow]",
            border_style="yellow"
        ))
    
    return column, value 
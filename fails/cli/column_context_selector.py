#!/usr/bin/env python3
"""
Simple arrow key selector that works reliably across different terminals.
"""

import sys
import termios
import tty
from typing import List, Set, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.text import Text


class SimpleArrowSelector:
    """Simple column selector with arrow keys that avoids terminal issues."""
    
    def __init__(self, columns: List[str], preselected: Set[str]):
        self.columns = sorted(columns)
        self.selected = preselected.copy()
        self.current_index = 0
        
        # Group columns
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
        
        # Build flat list with group headers
        for group in sorted(grouped.keys()):
            # Capitalize first letter for display
            display_name = group.capitalize()
            self.items.append(('group', display_name))
            for col in sorted(grouped[group]):
                self.items.append(('column', col))
        
        if other:
            self.items.append(('group', "Other"))
            for col in sorted(other):
                self.items.append(('column', col))
        
        # Ensure we start on a column, not a group header
        while self.current_index < len(self.items) and self.items[self.current_index][0] == 'group':
            self.current_index += 1
    
    def run(self) -> Set[str]:
        """Run the selector with simple display."""
        # Save terminal settings
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        
        try:
            # Enter alternate screen buffer and hide cursor
            sys.stdout.write("\033[?1049h\033[?25l")
            sys.stdout.flush()
            
            # Set terminal to raw mode
            tty.setraw(sys.stdin.fileno())
            
            # Initial clear of the alternate screen
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()
            
            while True:
                # Move to home and clear each line as we write
                sys.stdout.write("\033[H")
                
                # Header with rounded box style
                sys.stdout.write("\033[K\r\n")
                sys.stdout.write("\033[K\r\n")
                sys.stdout.write("\033[K  ╭" + "─" * 92 + "╮\r\n")
                sys.stdout.write("\033[K  │  \033[1m📋 Column Selection\033[0m" + " " * 72 + "│\r\n")
                sys.stdout.write("\033[K  │" + " " * 92 + "│\r\n")
                sys.stdout.write("\033[K  │  Select columns to include in the output. " + f"Selected: {len(self.selected)}/{len(self.columns)}" + " " * (48 - len(f"Selected: {len(self.selected)}/{len(self.columns)}")) + "│\r\n")
                sys.stdout.write("\033[K  │" + " " * 92 + "│\r\n")
                sys.stdout.write("\033[K  │  \033[2m↑/↓: Navigate    Space: Select    a: All    n: None    q: Save\033[0m" + " " * 28 + "│\r\n")
                sys.stdout.write("\033[K  ╰" + "─" * 92 + "╯\r\n")
                sys.stdout.write("\033[K\r\n")
                sys.stdout.write("\033[K\r\n")
                
                # Calculate window (show 20 items)
                window_size = 20
                half_window = window_size // 2
                
                # Calculate start position
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
                        # Add empty line before group (except for first item or when at start of window)
                        if i > 0 and i > start:
                            sys.stdout.write("\033[K\r\n")
                        # Group headers are never selectable, always use same prefix
                        # Use bold yellow for group headers to match failure_selector
                        sys.stdout.write(f"\033[K   \033[1;33m{item_data}\033[0m\r\n")
                    else:
                        # Column
                        is_selected = item_data in self.selected
                        
                        if is_current:
                            prefix = " ▶ "
                            # Highlight current selection in cyan
                            sys.stdout.write(f"\033[K{prefix} \033[96m{item_data}\033[0m\r\n")
                        else:
                            prefix = "   "
                            # Show selected items in a different color
                            if is_selected:
                                sys.stdout.write(f"\033[K{prefix}   \033[92m{item_data}\033[0m\r\n")  # Green for selected
                            else:
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
                    continue
                
                # Handle other keys
                if key == 'q':
                    break
                elif key == ' ' or key == '\r' or key == '\n':  # Space or Enter - toggle
                    if self.current_index < len(self.items) and self.items[self.current_index][0] == 'column':
                        col = self.items[self.current_index][1]
                        if col in self.selected:
                            self.selected.remove(col)
                        else:
                            self.selected.add(col)
                elif key == 'a':  # Select all
                    for item_type, item_data in self.items:
                        if item_type == 'column':
                            self.selected.add(item_data)
                elif key == 'n':  # Select none
                    self.selected.clear()
                elif key in ['j', 'k']:  # Vim keys
                    if key == 'j':
                        self.current_index = min(len(self.items) - 1, self.current_index + 1)
                        # Skip group headers when navigating
                        while (self.current_index < len(self.items) - 1 and 
                               self.items[self.current_index][0] == 'group'):
                            self.current_index += 1
                    else:
                        self.current_index = max(0, self.current_index - 1)
                        # Skip group headers when navigating
                        while (self.current_index > 0 and 
                               self.items[self.current_index][0] == 'group'):
                            self.current_index -= 1
                    
        finally:
            # Restore terminal
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            # Show cursor and exit alternate screen
            sys.stdout.write("\033[?25h\033[?1049l")
            sys.stdout.flush()
        
        return self.selected


def simple_arrow_selection(
    console: Console,
    columns: List[str],
    preselected: Set[str]
) -> Set[str]:
    """
    Simple arrow key selection that works reliably.
    
    Uses basic terminal control without Rich for the interactive part.
    """
    # Show initial message
    console.print(Panel(
        "[bold cyan]Column Selection[/bold cyan]\n\n"
        "Starting arrow key selector...\n"
        "[dim]Use ↑/↓ to navigate, Space to toggle[/dim]",
        border_style="cyan"
    ))
    
    # Run the selector
    selector = SimpleArrowSelector(columns, preselected)
    selected = selector.run()
    
    # Show results with Rich
    console.print(Panel(
        f"[bold green]Selection Complete![/bold green]\n\n"
        f"Selected {len(selected)} out of {len(columns)} columns",
        border_style="green"
    ))
    
    if selected:
        console.print("\n[bold]Selected columns:[/bold]")
        
        # Group for display
        grouped = {}
        other = []
        
        for col in sorted(selected):
            if '.' in col:
                prefix = col.split('.')[0]
                if prefix not in grouped:
                    grouped[prefix] = []
                grouped[prefix].append(col)
            else:
                other.append(col)
        
        for group, cols in sorted(grouped.items()):
            console.print(f"\n[yellow]{group}:[/yellow]")
            for col in cols:
                console.print(f"  • {col}")
        
        if other:
            console.print(f"\n[yellow]other:[/yellow]")
            for col in other:
                console.print(f"  • {col}")
    
    return selected


# Test function
def test_simple_arrow():
    """Test the simple arrow selector."""
    console = Console()
    
    columns = [
        "id", "name", "status",
        "user.id", "user.name", "user.email",
        "output.result", "output.scores",
        "metrics.latency", "metrics.cost"
    ]
    
    preselected = {"id", "user.name", "output.result"}
    
    selected = simple_arrow_selection(console, columns, preselected)
    
    console.print(f"\n[green]Final selection: {selected}[/green]")


if __name__ == "__main__":
    test_simple_arrow()
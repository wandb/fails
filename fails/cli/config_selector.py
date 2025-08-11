#!/usr/bin/env python3
"""
Interactive selector for choosing a configuration file.
Uses raw terminal mode for arrow-based selection, matching other selectors in the app.
"""

import sys
import tty
import termios
from pathlib import Path
from typing import List, Optional, Tuple
import yaml
from fails.cli.header import get_fails_header_for_raw_terminal


class ConfigSelector:
    """Interactive configuration file selector using arrow keys."""
    
    def __init__(self, config_dir: str = "./config"):
        """Initialize the config selector.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.configs = self._find_configs()
        self.current_index = 0
        
    def _find_configs(self) -> List[Tuple[str, dict]]:
        """Find all valid configuration files.
        
        Returns:
            List of tuples containing (filepath, config_data)
        """
        configs = []
        
        if not self.config_dir.exists():
            return configs
            
        for config_file in self.config_dir.glob("*.yaml"):
            # Skip test and eval configs
            if config_file.stem.lower().startswith(("test_", "fails-eval_")):
                continue
                
            try:
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                    
                # Extract entity and project from filename or config
                filename = config_file.stem  # Remove .yaml extension
                parts = filename.split('_')
                
                # Try to parse entity_project format
                if len(parts) >= 2:
                    # Join all parts except the last one as entity (handles underscores in entity names)
                    if parts[-1] == "config":
                        parts = parts[:-1]  # Remove 'config' suffix if present
                    
                    # Find the split between entity and project
                    # Assume last part is project, rest is entity
                    entity = "_".join(parts[:-1]) if len(parts) > 1 else parts[0]
                    project = parts[-1] if len(parts) > 1 else "unknown"
                else:
                    entity = "unknown"
                    project = filename
                
                config_info = {
                    'filepath': str(config_file),
                    'filename': config_file.name,
                    'entity': entity,
                    'project': project,
                    'data': config_data
                }
                configs.append((str(config_file), config_info))
                
            except Exception:
                # Skip invalid config files
                continue
                
        return sorted(configs, key=lambda x: x[1]['filename'])
    
    def _clear_screen(self):
        """Clear from current position down."""
        # Don't clear the entire screen, just from cursor down
        sys.stdout.write("\033[J")
        sys.stdout.flush()
        
    def display(self):
        """Display the configuration selection interface."""
        # Clear screen and move to home position
        sys.stdout.write("\033[H\033[J")
        
        # Add breathing room at the top
        sys.stdout.write("\r\n\r\n")
        
        # Print FAILS header using shared module
        sys.stdout.write(get_fails_header_for_raw_terminal())
        sys.stdout.write("\r\n")
        
        # Configuration selector header
        sys.stdout.write("\033[1m\033[96mSelect Evaluation Configuration \033[0m\r\n")
        sys.stdout.write("\r\n")
        
        if not self.configs:
            sys.stdout.write("\033[91mNo configuration files found in ./config/\033[0m\r\n")
            sys.stdout.write("\033[93mPlease run with --force-eval-select to create a new configuration.\033[0m\r\n")
        else:
            sys.stdout.write(f"Found \033[95m{len(self.configs)}\033[0m evaluation config(s), please select one or create a new config:\r\n")
            sys.stdout.write("\r\n")

            # Instructions (matching other selectors' style)
            sys.stdout.write("\033[2m─────────────────────────────────────────────────────────────────────────────────────────\033[0m\r\n")
            sys.stdout.write("\033[2m↑/↓: Navigate    Enter: Select    q: Quit    n: New config\033[0m\r\n")
            sys.stdout.write("\033[2m─────────────────────────────────────────────────────────────────────────────────────────\033[0m\r\n")
            sys.stdout.write("\r\n")

            # Display each config
            for i, (_, info) in enumerate(self.configs):  # _ to ignore unused filepath
                is_current = i == self.current_index
                
                if is_current:
                    # Highlight current selection in cyan
                    prefix = " \033[96m▶\033[0m "
                    sys.stdout.write(f"{prefix} \033[96m{info['entity']}/{info['project']}\033[0m\r\n")
                    
                    # Show details for selected config (indented)
                    if info['data']:
                        # Check for saved columns
                        first_key = list(info['data'].keys())[0] if info['data'] else None
                        if first_key and isinstance(info['data'][first_key], dict):
                            config_details = info['data'][first_key]
                            
                            if 'selected_columns' in config_details:
                                cols = config_details['selected_columns']
                                sys.stdout.write(f"     \033[2mColumns: {len(cols)} selected\033[0m\r\n")
                            
                            if 'failure_column' in config_details:
                                sys.stdout.write(f"     \033[2mFailure column name: {config_details['failure_column']}\033[0m\r\n")
                else:
                    prefix = "   "
                    sys.stdout.write(f"{prefix} {info['entity']}/{info['project']}\r\n")
            
            sys.stdout.write("\r\n")
        
        sys.stdout.flush()
        
    def run(self) -> Optional[dict]:
        """Run the configuration selector.
        
        Returns:
            Selected configuration info dict or None if cancelled/no configs
        """
        if not self.configs:
            # No configs available, return signal to create new config
            self._clear_screen()
            self.display()
            print("\n\033[93mNo saved configurations found. Starting new configuration setup...\033[0m")
            return {'force_selection': True, 'no_configs': True}
            
        # Save terminal settings
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        
        try:
            # Enter alternate screen buffer and hide cursor
            sys.stdout.write("\033[?1049h\033[?25l")
            sys.stdout.flush()
            
            # Set terminal to raw mode (use direct fileno call for better compatibility)
            tty.setraw(sys.stdin.fileno())
            
            # Initial clear of the alternate screen
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()
            
            while True:
                self.display()
                
                # Read key
                key = sys.stdin.read(1)
                
                if key == '\x1b':  # Escape sequence
                    seq = sys.stdin.read(2)
                    # Handle both bracket and O formats for arrow keys (zsh/iTerm compatibility)
                    if seq == '[A' or seq == 'OA':  # Up arrow (both formats)
                        if self.current_index > 0:
                            self.current_index -= 1
                    elif seq == '[B' or seq == 'OB':  # Down arrow (both formats)
                        if self.current_index < len(self.configs) - 1:
                            self.current_index += 1
                            
                elif key in ('q', 'Q', '\x03'):  # q or Ctrl+C
                    return None
                    
                elif key in ('n', 'N'):  # New config
                    # Return special marker to trigger force selection
                    return {'force_selection': True}
                    
                elif key == '\r':  # Enter
                    if self.configs:
                        _, selected_info = self.configs[self.current_index]
                        return selected_info
                        
        finally:
            # Restore terminal
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            sys.stdout.write("\033[?25h\033[?1049l")  # Show cursor and exit alternate screen
            sys.stdout.flush()


def select_config(config_dir: str = "./config") -> Optional[dict]:
    """Interactive configuration selection.
    
    Args:
        config_dir: Directory containing configuration files
        
    Returns:
        Selected configuration info or None if cancelled
    """
    selector = ConfigSelector(config_dir)
    result = selector.run()
    
    if result and not result.get('force_selection'):
        # Print confirmation
        print(f"\n\033[95m✓\033[0m Selected configuration: \033[96m{result['entity']}/{result['project']}\033[0m")
        
    return result


if __name__ == "__main__":
    # Test the selector
    config = select_config()
    if config:
        if config.get('force_selection'):
            print("User requested new configuration setup")
        else:
            print(f"Selected: {config['filepath']}")
    else:
        print("No configuration selected")
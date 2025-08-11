#!/usr/bin/env python3
"""
Interactive selector for entering a Weave Evaluation ID or URL.
Uses prompt_toolkit for robust paste handling and clean UI.
"""

import re
from typing import Optional
from prompt_toolkit import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit, Window, ConditionalContainer, VSplit
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.widgets import Frame, TextArea
from prompt_toolkit.filters import Condition
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import FormattedText


class UserCancelledException(Exception):
    """Exception raised when user cancels the selection."""


class EvaluationSelector:
    """Simple selector for entering a Weave Evaluation URL."""
    
    # Regex patterns for extracting components from URL
    # Example URL: https://wandb.ai/wandb-applied-ai-team/eval-failures/weave/calls/01985cfc-6d70-711f-89b7-bb22c693ca75
    # Pattern to match both direct calls and evaluation URLs with peekPath
    URL_PATTERN = r'https?://wandb\.ai/([^/]+)/([^/]+)/weave/(?:calls/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})|evaluations\?.*peekPath=%2F[^%]+%2F[^%]+%2Fcalls%2F([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}))'
    
    # Legacy patterns for backward compatibility
    ID_PATTERNS = [
        # Pattern for URL with /calls/ path
        r'/calls/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})',
        # Pattern for evaluation ID in query params or path
        r'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})',
    ]
    
    def __init__(self, default_value: Optional[str] = None):
        self.default_value = default_value
        self.example_url = "https://wandb.ai/your-entity/your-project/weave/calls/01985cfc-6d70-711f-89b7-bb22c693ca75"
        self.result = None  # Will be tuple of (entity, project, eval_id)
        self.error_message = ""
        self.MAX_INPUT_SIZE = 500
    
    def extract_components_from_url(self, url: str) -> Optional[tuple]:
        """Extract entity, project, and evaluation ID from a Weave URL.
        
        Returns:
            Tuple of (entity, project, eval_id) or None if extraction fails
        """
        # Try to match the full URL pattern
        match = re.search(self.URL_PATTERN, url)
        if match:
            entity = match.group(1)
            project = match.group(2)
            # Group 3 is for direct calls, group 4 is for evaluation peekPath
            eval_id = match.group(3) or match.group(4)
            return (entity, project, eval_id)
        
        # Fallback: try to extract just the ID for backward compatibility
        for pattern in self.ID_PATTERNS:
            match = re.search(pattern, url)
            if match:
                # Return None for entity/project if we can only extract ID
                return (None, None, match.group(1))
        
        return None
    
    def validate_id(self, eval_id: str) -> bool:
        """Validate that a string looks like an evaluation ID."""
        # Check for UUID-like format (8-4-4-4-12 hex characters)
        pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        return bool(re.match(pattern, eval_id.lower()))
    
    def extract_safely(self, input_str: str) -> Optional[tuple]:
        """Try to extract entity, project, and evaluation ID from input.
        
        Returns:
            Tuple of (entity, project, eval_id) or None if extraction fails
        """
        # Limit how much we search through
        if len(input_str) > 5000:
            input_str = input_str[:5000]
        
        clean_input = input_str.strip()
        
        # Try to extract from URL
        result = self.extract_components_from_url(clean_input)
        if result:
            return result
        
        # For backward compatibility: check if it's a direct ID
        if self.validate_id(clean_input):
            return (None, None, clean_input)
        
        return None
    
    def run(self) -> Optional[tuple]:
        """Run the selector using prompt_toolkit and return (entity, project, eval_id)."""
        
        # Create the text input area with multiline support
        text_area = TextArea(
            text=self.default_value or "",
            multiline=True,  # Allow multiline for display
            wrap_lines=True,  # Wrap long lines
            scrollbar=False,
            focus_on_click=True,
            height=3,  # Allow up to 3 lines of display
        )
        
        # Create header text with formatted instructions  
        header_text = FormattedText([
            ('', '\n'),
            ('', 'Please provide a Weave Evaluation URL.\n'),
            ('', '\n'),
            ('', 'The URL should look like:\n'),
            ('', f'  {self.example_url}...\n'),
            ('', '\n'),
            ('class:instructions', 'Hint: Click on the Evaluation row in the Evals tab, then copy the URL.\n'),
            ('', '\n'),
            ('', '---------------------------------------------------------------------------\n'),
            ('class:instructions', "Type 'q' to quit |  Enter to submit |  Ctrl+U to clear |  Ctrl+C to cancel\n"),
            ('', '---------------------------------------------------------------------------\n'),
        ])
        
        # Create a window for error messages
        def get_error_text():
            return self.error_message if self.error_message else ""
        
        error_control = FormattedTextControl(
            text=get_error_text,
            focusable=False
        )
        
        # Create the input frame with border
        input_frame = Frame(
            text_area,
            title="Enter Evaluation URL" + (f" [{self.default_value}]" if self.default_value else ""),
            width=75,  # Increased width for longer URLs
            height=5,  # Increased to accommodate 3 lines of text plus border
        )
        
        # Left-align the input frame
        left_aligned_input = VSplit([
            input_frame,
            Window(),  # Right padding (expands to fill)
        ])
        
        # Create the layout
        root_container = HSplit([
            Window(FormattedTextControl(text=header_text), height=15),
            Window(height=1),  # Add breathing room above the box
            left_aligned_input,
            ConditionalContainer(
                Window(error_control, height=3, style="class:error"),
                filter=Condition(lambda: bool(self.error_message))
            ),
        ])
        
        # Define key bindings
        kb = KeyBindings()
        
        @kb.add('c-c')
        def _(event):
            """Exit on Ctrl+C."""
            event.app.exit(result=None)
        
        @kb.add('c-u')
        def _(event):
            """Clear input on Ctrl+U."""
            text_area.text = ""
            self.error_message = ""
        
        @kb.add('enter', filter=True)  # Always capture Enter
        def _(event):
            """Process input on Enter - submit the form."""
            user_input = text_area.text.strip()
            
            # Check for quit commands
            if user_input.lower() in ('q', 'quit', 'exit'):
                event.app.exit(result=None)
                return
            
            # Use default if empty and available
            if not user_input and self.default_value:
                user_input = self.default_value
            
            # Empty input
            if not user_input:
                self.error_message = "❌ Please enter a Weave evaluation URL."
                return
            
            # Check input length
            if len(user_input) > self.MAX_INPUT_SIZE:
                self.error_message = f"❌ Input too long (>{self.MAX_INPUT_SIZE} chars). Please paste a single evaluation URL."
                text_area.text = ""
                return
            
            # Try to extract components
            result = self.extract_safely(user_input)
            if result:
                event.app.exit(result=result)
            else:
                # Provide specific feedback
                if 'wandb.ai' in user_input.lower():
                    self.error_message = "❌ Could not extract information from URL. Please ensure it's a valid Weave evaluation URL."
                else:
                    self.error_message = "❌ Please provide a valid Weave evaluation URL from W&B."
        
        # Monitor text changes to check for paste size
        def on_text_changed(_):
            """Check text size on every change."""
            if len(text_area.text) > self.MAX_INPUT_SIZE:
                # Clear the input and show error
                text_area.text = ""
                self.error_message = f"❌ Input too long. Please paste a single evaluation ID or URL (max {self.MAX_INPUT_SIZE} chars)."
            else:
                # Clear error message when typing normally
                if self.error_message and len(text_area.text) < self.MAX_INPUT_SIZE:
                    self.error_message = ""
        
        text_area.buffer.on_text_changed += on_text_changed
        
        # Define custom style
        style = Style.from_dict({
            'frame.border': '#888888',
            'error': 'fg:red bold',
            'instructions': 'fg:#888888',  # Grey color for instructions
            'step_header': 'fg:cyan bold',  # Cyan bold for step header
        })
        
        # Create and run the application
        app = Application(
            layout=Layout(root_container),
            key_bindings=kb,
            style=style,
            full_screen=False,
            mouse_support=True,
        )
        
        # Run the application and get the result
        result = app.run()
        
        # Print confirmation if we got valid results
        if result:
            entity, project, eval_id = result
            if entity and project:
                print(f"\n\033[95m✓\033[0m Extracted from URL:")
                print(f"  Entity: \033[96m{entity}\033[0m")
                print(f"  Project: \033[96m{project}\033[0m")
                print(f"  Evaluation ID: \033[96m{eval_id}\033[0m")
            else:
                print(f"\n\033[95m✓\033[0m Valid Weave Evaluation ID extracted: \033[96m{eval_id}\033[0m")
        
        return result


def interactive_evaluation_selection(console=None, default_value: Optional[str] = None) -> Optional[tuple]:
    """
    Interactive function to select a Weave evaluation from URL.
    
    Args:
        console: Optional Rich console (not used, kept for compatibility)
        default_value: Optional default evaluation URL
        
    Returns:
        Tuple of (entity, project, eval_id) or None if cancelled
    """
    # If console is a string, treat it as the default_value for backward compatibility
    if isinstance(console, str):
        default_value = console
        console = None
    
    try:
        selector = EvaluationSelector(default_value=default_value)
        return selector.run()
    except KeyboardInterrupt:
        return None
    except Exception as e:
        print(f"Error during evaluation selection: {e}")
        return None


def main():
    """Test the selector."""
    try:
        result = interactive_evaluation_selection()
        
        if result:
            entity, project, eval_id = result
            print("Selected:")
            if entity and project:
                print(f"  Entity: {entity}")
                print(f"  Project: {project}")
            print(f"  Evaluation ID: {eval_id}")
        else:
            print("No evaluation selected.")
    except KeyboardInterrupt:
        print("\nCancelled.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()